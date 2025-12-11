import React, { useRef, useEffect, useCallback } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import type { NetworkTopology, NetworkConnection } from '../../types/NetworkTopologyTypes';
import type { NodeDTO } from '../../types/NodeTypes';
import type { UserTerminal } from '../../types/UserTerminalTypes';

interface NetworkTopologyViewProps {
    topology: NetworkTopology | null;
    selectedNodeIds?: Set<string>; // Filter: only show selected nodes
    onNodeClick?: (node: NodeDTO) => void;
    onTerminalClick?: (terminal: UserTerminal) => void;
}

const NetworkTopologyView: React.FC<NetworkTopologyViewProps> = ({ topology, selectedNodeIds, onNodeClick, onTerminalClick }) => {
    const cesiumContainer = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<Cesium.Viewer | null>(null);
    const nodeEntitiesRef = useRef<Map<string, Cesium.Entity>>(new Map());
    const terminalEntitiesRef = useRef<Map<string, Cesium.Entity>>(new Map());
    const connectionLinesRef = useRef<Map<string, Cesium.Entity>>(new Map());

    // Initialize Cesium (only once)
    useEffect(() => {
        if (cesiumContainer.current && !viewerRef.current) {
            try {
                const viewer = new Cesium.Viewer(cesiumContainer.current, {
                    timeline: false,
                    animation: false,
                    baseLayerPicker: false,
                    geocoder: false,
                    homeButton: true,
                    sceneModePicker: true,
                    navigationHelpButton: false,
                    infoBox: false,
                    selectionIndicator: false,
                });
                viewer.scene.requestRenderMode = true;
                viewer.scene.globe.depthTestAgainstTerrain = true;
                viewer.scene.globe.enableLighting = false;
                viewer.scene.fog.enabled = false;
                viewer.scene.globe.baseColor = Cesium.Color.DARKSLATEGRAY;
                viewer.scene.backgroundColor = Cesium.Color.BLACK;
                viewer.camera.setView({
                    destination: Cesium.Cartesian3.fromDegrees(0, 0, 2e7),
                });

                viewerRef.current = viewer;
            } catch (error) {
                console.error('Failed to initialize Cesium viewer:', error);
            }
        }

        return () => {
            // Cleanup on unmount
            if (viewerRef.current && !viewerRef.current.isDestroyed()) {
                try {
                    viewerRef.current.destroy();
                } catch (error) {
                    console.error('Error destroying Cesium viewer:', error);
                }
                viewerRef.current = null;
            }
        };
    }, []); // Only run once on mount

    // Get node warning color based on metrics
    const getNodeWarningColor = (node: NodeDTO): Cesium.Color => {
        // Check for critical issues (red)
        if (node.nodeProcessingDelayMs > 500 || 
            node.packetLossRate > 0.1 || 
            (node.currentPacketCount / node.packetBufferCapacity) > 0.9 ||
            node.resourceUtilization > 90) {
            return Cesium.Color.RED;
        }
        
        // Check for warnings (yellow)
        if (node.nodeProcessingDelayMs > 200 || 
            node.packetLossRate > 0.05 || 
            (node.currentPacketCount / node.packetBufferCapacity) > 0.7 ||
            node.resourceUtilization > 70) {
            return Cesium.Color.YELLOW;
        }
        
        // Normal (green) - but keep type-based color for visibility
        return Cesium.Color.LIME;
    };

    // Get node color based on type (fallback)
    const getNodeTypeColor = (nodeType: string): Cesium.Color => {
        if (nodeType.includes('SATELLITE')) {
            if (nodeType.includes('LEO')) return Cesium.Color.CYAN;
            if (nodeType.includes('MEO')) return Cesium.Color.BLUE;
            if (nodeType.includes('GEO')) return Cesium.Color.PURPLE;
            return Cesium.Color.CYAN;
        }
        return Cesium.Color.ORANGE;
    };

    // Get connection color based on status
    const getConnectionColor = (status: string): Cesium.Color => {
        switch (status) {
            case 'active':
                return Cesium.Color.LIME;
            case 'degraded':
                return Cesium.Color.YELLOW;
            case 'inactive':
                return Cesium.Color.RED;
            default:
                return Cesium.Color.GRAY;
        }
    };

    // Add node to map
    const addNode = useCallback((node: NodeDTO) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        // Filter: skip if not in selectedNodeIds (if filter is active)
        if (selectedNodeIds && selectedNodeIds.size > 0 && !selectedNodeIds.has(node.nodeId)) {
            return;
        }

        const lon = Number(node.position?.longitude);
        const lat = Number(node.position?.latitude);
        if (isNaN(lon) || isNaN(lat)) return;

        const isSatellite = node.nodeType.includes('SATELLITE');
        const warningColor = getNodeWarningColor(node);
        const typeColor = getNodeTypeColor(node.nodeType);
        // Use warning color if there's an issue, otherwise use type color
        const finalColor = warningColor.equals(Cesium.Color.LIME) ? typeColor : warningColor;
        
        const position = Cesium.Cartesian3.fromDegrees(lon, lat, node.position?.altitude || 0);

        // Build status text
        const warnings: string[] = [];
        if (node.nodeProcessingDelayMs > 200) warnings.push(`Latency: ${node.nodeProcessingDelayMs}ms`);
        if (node.packetLossRate > 0.05) warnings.push(`Loss: ${(node.packetLossRate * 100).toFixed(1)}%`);
        const queueRatio = node.packetBufferCapacity > 0 ? (node.currentPacketCount / node.packetBufferCapacity) * 100 : 0;
        if (queueRatio > 70) warnings.push(`Queue: ${queueRatio.toFixed(0)}%`);
        if (node.resourceUtilization > 70) warnings.push(`Util: ${node.resourceUtilization}%`);
        
        const statusText = warnings.length > 0 ? `⚠️ ${warnings.join(', ')}` : node.nodeName;

        const entity = viewer.entities.add({
            id: node.nodeId,
            name: node.nodeName,
            position: position,
            point: {
                pixelSize: isSatellite ? 14 : 12, // Slightly larger for visibility
                color: finalColor,
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: warnings.length > 0 ? 3 : 2, // Thicker outline for warnings
                heightReference: isSatellite ? Cesium.HeightReference.NONE : Cesium.HeightReference.CLAMP_TO_GROUND,
            },
            label: {
                text: statusText,
                font: warnings.length > 0 ? 'bold 12px sans-serif' : '12px sans-serif',
                fillColor: finalColor,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                pixelOffset: new Cesium.Cartesian2(0, -25),
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            },
        });

        nodeEntitiesRef.current.set(node.nodeId, entity);
    }, [selectedNodeIds]);

    // Add terminal to map
    const addTerminal = useCallback((terminal: UserTerminal) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        const lon = Number(terminal.position?.longitude);
        const lat = Number(terminal.position?.latitude);
        if (isNaN(lon) || isNaN(lat)) return;

        const statusColor = terminal.status === 'connected' ? Cesium.Color.GREEN : Cesium.Color.GRAY;
        const position = Cesium.Cartesian3.fromDegrees(lon, lat, terminal.position?.altitude || 0);

        const entity = viewer.entities.add({
            id: `TERMINAL-${terminal.terminalId}`,
            name: terminal.terminalName,
            position: position,
            point: {
                pixelSize: 8,
                color: statusColor,
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 1,
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
            },
        });

        terminalEntitiesRef.current.set(terminal.terminalId, entity);
    }, []);

    // Add connection line
    const addConnection = useCallback((connection: NetworkConnection, nodeMap: Map<string, NodeDTO>) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        const fromNode = nodeMap.get(connection.fromNodeId);
        const toNode = nodeMap.get(connection.toNodeId);

        if (!fromNode || !toNode) return;

        const fromPos = Cesium.Cartesian3.fromDegrees(
            fromNode.position.longitude,
            fromNode.position.latitude,
            fromNode.position.altitude || 0
        );
        const toPos = Cesium.Cartesian3.fromDegrees(
            toNode.position.longitude,
            toNode.position.latitude,
            toNode.position.altitude || 0
        );

        // Calculate distance to ensure valid polyline
        const distance = Cesium.Cartesian3.distance(fromPos, toPos);
        if (distance < 100) {
            // Skip very short connections to avoid Cesium errors
            return;
        }

        const color = getConnectionColor(connection.status);
        const lineId = `CONN-${connection.fromNodeId}-${connection.toNodeId}`;

        try {
            // Ensure minimum width for Cesium (must be >= 0.0125)
            const lineWidth = Math.max(0.02, connection.status === 'active' ? 3.0 : 2.0);
            
            const entity = viewer.entities.add({
                id: lineId,
                polyline: {
                    positions: [fromPos, toPos],
                    width: lineWidth,
                    material: color.withAlpha(0.6),
                    clampToGround: false,
                    arcType: Cesium.ArcType.GEODESIC,
                },
            });

            connectionLinesRef.current.set(lineId, entity);
        } catch (error) {
            console.warn(`Failed to add connection line ${lineId}:`, error);
        }
    }, []);

    // Update topology visualization
    useEffect(() => {
        if (!viewerRef.current || !topology || viewerRef.current.isDestroyed()) return;
        const viewer = viewerRef.current;

        try {
            // Clear existing entities
            nodeEntitiesRef.current.forEach((entity) => {
                try {
                    viewer.entities.remove(entity);
                } catch (e) {
                    // Entity may already be removed
                }
            });
            terminalEntitiesRef.current.forEach((entity) => {
                try {
                    viewer.entities.remove(entity);
                } catch (e) {
                    // Entity may already be removed
                }
            });
            connectionLinesRef.current.forEach((entity) => {
                try {
                    viewer.entities.remove(entity);
                } catch (e) {
                    // Entity may already be removed
                }
            });
            nodeEntitiesRef.current.clear();
            terminalEntitiesRef.current.clear();
            connectionLinesRef.current.clear();
        } catch (error) {
            console.error('Error clearing entities:', error);
            return;
        }

        // Create node map for quick lookup
        const nodeMap = new Map<string, NodeDTO>();
        topology.nodes.forEach((node) => {
            nodeMap.set(node.nodeId, node);
            addNode(node);
        });

        // Only show connections between visible nodes
        const visibleNodeIds = new Set(Array.from(nodeEntitiesRef.current.keys()));

        // Add terminals
        topology.terminals.forEach((terminal) => {
            addTerminal(terminal);
        });

        // Add connections (only between visible nodes)
        try {
            topology.connections.forEach((connection) => {
                if (visibleNodeIds.has(connection.fromNodeId) && visibleNodeIds.has(connection.toNodeId)) {
                    addConnection(connection, nodeMap);
                }
            });
        } catch (error) {
            console.error('Error adding connections:', error);
        }

        // Set click handlers
        const clickHandler = (event: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
            const picked = viewer.scene.pick(event.position);
            if (picked?.id) {
                const id = picked.id.id as string;
                
                if (id.startsWith('TERMINAL-')) {
                    const terminalId = id.replace('TERMINAL-', '');
                    const terminal = topology.terminals.find(t => t.terminalId === terminalId);
                    if (terminal && onTerminalClick) {
                        onTerminalClick(terminal);
                    }
                } else {
                    const node = topology.nodes.find(n => n.nodeId === id);
                    if (node && onNodeClick) {
                        onNodeClick(node);
                    }
                }
            }
        };

        viewer.screenSpaceEventHandler.setInputAction(
            clickHandler,
            Cesium.ScreenSpaceEventType.LEFT_CLICK
        );

        return () => {
            if (viewerRef.current && !viewerRef.current.isDestroyed()) {
                viewerRef.current.screenSpaceEventHandler.removeInputAction(
                    Cesium.ScreenSpaceEventType.LEFT_CLICK
                );
            }
        };
    }, [topology, selectedNodeIds, addNode, addTerminal, addConnection, onNodeClick, onTerminalClick]);

    return <div ref={cesiumContainer} className="w-full h-full" />;
};

export default NetworkTopologyView;


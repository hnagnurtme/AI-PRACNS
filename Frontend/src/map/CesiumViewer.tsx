// src/map/CesiumViewer.tsx
import React, { useEffect, useRef, useCallback } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { useNodeStore } from "../state/nodeStore"; // Đảm bảo import store đã cập nhật
import { useTerminalStore } from "../state/terminalStore";
import type { NodeDTO } from "../types/NodeTypes";
import type { UserTerminal } from "../types/UserTerminalTypes";
import type { RoutingPath, Packet } from "../types/RoutingTypes";
import { createTerminalFromMap, getUserTerminals } from '../services/userTerminalService';

import SATELLITEICON from "../assets/icons/SATELLITE.svg";
import STATIONICON from "../assets/icons/STATION.svg";

declare global {
    interface Window {
        viewer?: Cesium.Viewer;
    }
}

interface CesiumViewerProps {
    nodes: NodeDTO[];
    routingPath?: RoutingPath | null;
    activePackets?: Packet[]; // Packets to animate
    onClearPaths?: () => void; // Callback to clear all paths
    onPathClick?: (path: RoutingPath) => void; // Callback when path is clicked
    onTerminalCreated?: (terminal: UserTerminal) => void; // Callback when terminal is created
}

// ================== Constants ==================
const ORBIT_CONFIG = {
    LEO: { altitude: 550000, period: 5400 },
    MEO: { altitude: 20000000, period: 43200 },
    GEO: { altitude: 35786000, period: 86400 },
};

const VISUAL_CONFIG = {
    SATELLITE: {
        size: 24,
        scale: { near: 1.8, mid: 1.0, far: 0.1 },
    },
    STATION: {
        size: 32,  // Tăng từ 20 → 32 để dễ nhìn hơn
        scale: { near: 2.2, mid: 1.2, far: 0.15 },  // Tăng scale để nổi bật hơn
    },
    TERMINAL: {
        size: 24,
        scale: { near: 0.8, mid: 0.6, far: 0.2 }, 
        pointSize: 12, // Size cho point marker (chấm tròn) - giảm từ 16
    },
};

// ================== Utility ==================
const getNodeColor = (nodeType: string): Cesium.Color => {
    switch (nodeType) {
        case "LEO_SATELLITE": return Cesium.Color.CYAN;
        case "MEO_SATELLITE": return Cesium.Color.GOLD;
        case "GEO_SATELLITE": return Cesium.Color.ORANGE;
        case "GROUND_STATION": return Cesium.Color.YELLOW;  // Vàng sáng - nổi bật rõ ràng
        default: return Cesium.Color.WHITE;
    }
};

const getTerminalColor = (status: string): Cesium.Color => {
    switch (status) {
        case "idle": return Cesium.Color.DARKGRAY;           // Tối hơn để không lẫn với node
        case "connected": return Cesium.Color.LIMEGREEN;    // Xanh lá sáng - terminal đang kết nối
        case "transmitting": return Cesium.Color.GOLD;      // Vàng gold - đang truyền dữ liệu
        case "disconnected": return Cesium.Color.CRIMSON;   // Đỏ đậm - mất kết nối
        default: return Cesium.Color.WHITESMOKE;
    }
};

const getOrbitConfig = (nodeType: string) => {
    switch (nodeType) {
        case "LEO_SATELLITE": return ORBIT_CONFIG.LEO;
        case "MEO_SATELLITE": return ORBIT_CONFIG.MEO;
        case "GEO_SATELLITE": return ORBIT_CONFIG.GEO;
        default: return ORBIT_CONFIG.LEO;
    }
};

// ================== Tạo chuyển động tròn ==================
const createCircularOrbit = (
    node: NodeDTO,
    orbitConfig: { altitude: number; period: number }
): Cesium.SampledPositionProperty => {
    const positionProperty = new Cesium.SampledPositionProperty();
    const lon = Number(node.position?.longitude) || 0;
    const lat = Number(node.position?.latitude) || 0;
    const epoch = Cesium.JulianDate.now();
    const totalSamples = 720;
    const totalDuration = orbitConfig.period * 2;
    const step = totalDuration / totalSamples;
    const angularSpeed = (2 * Math.PI) / orbitConfig.period;

    for (let i = 0; i <= totalSamples; i++) {
        const timeOffset = i * step;
        const time = Cesium.JulianDate.addSeconds(epoch, timeOffset, new Cesium.JulianDate());
        const angle_rad = angularSpeed * timeOffset;
        const newLon = (lon + (angle_rad * 180 / Math.PI)) % 360;
        const position = Cesium.Cartesian3.fromDegrees(newLon, lat, orbitConfig.altitude);
        positionProperty.addSample(time, position);
    }
    positionProperty.setInterpolationOptions({
        interpolationDegree: 5,
        interpolationAlgorithm: Cesium.LagrangePolynomialApproximation,
    });
    return positionProperty;
};

// ================== Component ==================
// Algorithm color mapping
const getAlgorithmColor = (algorithm?: string): Cesium.Color => {
    switch (algorithm) {
        case 'simple':
            return Cesium.Color.LIGHTSEAGREEN;  // Xanh lục nhạt - simple routing
        case 'dijkstra':
            return Cesium.Color.DODGERBLUE;     // Xanh dương sáng - Dijkstra shortest path
        case 'rl':
            return Cesium.Color.MEDIUMPURPLE;   // Tím gradient - RL intelligent routing
        default:
            return Cesium.Color.LIGHTSEAGREEN;
    }
};

const CesiumViewer: React.FC<CesiumViewerProps> = ({ nodes, routingPath, activePackets = [], onClearPaths, onPathClick, onTerminalCreated }) => {
    const cesiumContainer = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<Cesium.Viewer | null>(null);
    const entityCacheRef = useRef<Map<string, Cesium.Entity>>(new Map());
    const terminalEntityCacheRef = useRef<Map<string, Cesium.Entity>>(new Map());
    const connectionLineCacheRef = useRef<Map<string, Cesium.Entity>>(new Map());
    const routingPathCacheRef = useRef<Map<string, Cesium.Entity>>(new Map());
    const packetPathCacheRef = useRef<Map<string, Cesium.Entity[]>>(new Map()); // Store multiple entities per packet path
    const pathDataCacheRef = useRef<Map<string, RoutingPath>>(new Map()); // Store path data for click handler
    // const packetAnimationsRef = useRef<Map<string, PacketAnimation>>(new Map()); // Disabled: no packet animation
    // const processedPacketIdsRef = useRef<Set<string>>(new Set()); // Disabled: no packet animation

    // [SỬA] Lấy 'flyToTrigger' từ store
    const { setSelectedNode, selectedNode, cameraFollowMode, flyToTrigger , setCameraFollowMode } = useNodeStore();
    const { terminals, setSelectedTerminal, sourceTerminal, destinationTerminal, setSourceTerminal, setDestinationTerminal, setTerminals } = useTerminalStore();
    
    // Thêm ref để theo dõi giá trị trigger (tránh lặp vô hạn)
    const lastTriggerRef = useRef(flyToTrigger);

    // ========== Tạo Billboard ==========
    const createBillboardOptions = useCallback((node: NodeDTO, isSatellite: boolean) => {
        const config = isSatellite ? VISUAL_CONFIG.SATELLITE : VISUAL_CONFIG.STATION;
        const color = getNodeColor(node.nodeType);
        return {
            image: isSatellite ? SATELLITEICON : STATIONICON,
            width: config.size,
            height: config.size,
            color,
            scaleByDistance: new Cesium.NearFarScalar(5e4, config.scale.near, 1e8, config.scale.far),
            pixelOffsetScaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.3),
            translucencyByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.6), // Ít trong suốt hơn để dễ nhìn
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 2e8),
            heightReference: isSatellite
                ? Cesium.HeightReference.NONE
                : Cesium.HeightReference.CLAMP_TO_GROUND,
        };
    }, []);

    // ========== Tạo Label (Improved) ==========
    const createLabelOptions = useCallback((node: NodeDTO) => {
        const color = getNodeColor(node.nodeType);
        const shortName = node.nodeName.length > 12 
            ? node.nodeName.substring(0, 10) + "…" 
            : node.nodeName;
        
        // Create a better styled label with background
        return new Cesium.LabelGraphics({
            text: shortName,
            font: 'bold 13px "Segoe UI", Arial, sans-serif',
            fillColor: Cesium.Color.WHITE,
            outlineColor: color,
            outlineWidth: 3,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            backgroundColor: color.withAlpha(0.8),
            backgroundPadding: new Cesium.Cartesian2(8, 5),
            pixelOffset: new Cesium.Cartesian2(0, -32),
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 3e6, 0.3),
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 3e6),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            translucencyByDistance: new Cesium.NearFarScalar(5e4, 1.0, 3e6, 0.5),
        });
    }, []);

    // ========== Thêm Node ==========
    const addNode = useCallback((node: NodeDTO) => {
        if (!viewerRef.current) return;
        const { nodeId, nodeType, position } = node;
        const lon = Number(position?.longitude);
        const lat = Number(position?.latitude);
        if (isNaN(lon) || isNaN(lat)) return;

        const isSatellite = nodeType.includes("SATELLITE");
        let positionProperty: Cesium.PositionProperty;

        if (isSatellite) {
            const orbitConfig = getOrbitConfig(nodeType);
            positionProperty = createCircularOrbit(node, orbitConfig);
        } else {
            const staticPos = Cesium.Cartesian3.fromDegrees(lon, lat, 100);
            positionProperty = new Cesium.ConstantPositionProperty(staticPos);
        }

        const existing = entityCacheRef.current.get(nodeId);
        if (existing) {
            viewerRef.current.entities.remove(existing);
        }

        const entity = viewerRef.current.entities.add({
            id: nodeId,
            name: node.nodeName,
            position: positionProperty,
            billboard: createBillboardOptions(node, isSatellite),
            label: createLabelOptions(node),
        });
        entityCacheRef.current.set(nodeId, entity);
    }, [createBillboardOptions, createLabelOptions]);

    // ========== Batch Add ==========
    const batchAddNodes = useCallback((nodesToAdd: NodeDTO[]) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        viewer.entities.suspendEvents();
        entityCacheRef.current.clear();
        nodesToAdd.forEach(addNode);
        viewer.entities.resumeEvents();
    }, [addNode]);

    // ========== Terminal Visualization ==========
    // Tạo icon chấm tròn cho terminals (giống GROUND_STATION nhưng khác icon)
    const createTerminalDotIcon = useCallback((color: Cesium.Color, size: number = 32): string => {
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        if (!ctx) return '';
        
        // Vẽ chấm tròn
        const centerX = size / 2;
        const centerY = size / 2;
        const radius = size / 2 - 2;
        
        // Vẽ outline
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(${color.red * 255}, ${color.green * 255}, ${color.blue * 255}, ${color.alpha})`;
        ctx.fill();
        
        // Vẽ border
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        return canvas.toDataURL();
    }, []);

    // Tạo billboard options cho terminals (giống GROUND_STATION)
    const createTerminalBillboard = useCallback((terminal: UserTerminal) => {
        const config = VISUAL_CONFIG.TERMINAL;
        const baseColor = getTerminalColor(terminal.status);
        
        // Highlight nếu terminal được chọn làm source hoặc destination
        let color = baseColor;
        let scale = config.scale.near;
        let size = config.size;
        
        if (sourceTerminal?.terminalId === terminal.terminalId) {
            color = Cesium.Color.SPRINGGREEN; // Source = xanh lá neon
            scale = config.scale.near * 1.2;
            size = config.size * 1.2;
        } else if (destinationTerminal?.terminalId === terminal.terminalId) {
            color = Cesium.Color.ORANGERED; // Destination = đỏ cam
            scale = config.scale.near * 1.2;
            size = config.size * 1.2;
        }
        
        // Tạo icon chấm tròn
        const iconDataUrl = createTerminalDotIcon(color, size);
        
        return {
            image: iconDataUrl,
            width: size,
            height: size,
            scale: scale / config.scale.near,
            scaleByDistance: new Cesium.NearFarScalar(5e4, config.scale.near, 1e8, config.scale.far),
            pixelOffsetScaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.3),
            translucencyByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.6),
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 2e8),
            heightReference: Cesium.HeightReference.CLAMP_TO_GROUND, // Gắn vào mặt đất giống GROUND_STATION
        };
    }, [sourceTerminal, destinationTerminal, createTerminalDotIcon]);

    const createTerminalLabel = useCallback((terminal: UserTerminal) => {
        const baseColor = getTerminalColor(terminal.status);
        
        // Highlight label nếu terminal được chọn
        let fillColor = baseColor;
        let backgroundColor = baseColor.withAlpha(0.8);
        
        if (sourceTerminal?.terminalId === terminal.terminalId) {
            fillColor = Cesium.Color.SPRINGGREEN;
            backgroundColor = Cesium.Color.SPRINGGREEN.withAlpha(0.9);
        } else if (destinationTerminal?.terminalId === terminal.terminalId) {
            fillColor = Cesium.Color.ORANGERED;
            backgroundColor = Cesium.Color.ORANGERED.withAlpha(0.9);
        }
        
        const shortName = terminal.terminalName.length > 12 
            ? terminal.terminalName.substring(0, 10) + "…" 
            : terminal.terminalName;
        // GIỐNG HỆT GROUND_STATION: không có heightReference trong label
        // Label sẽ tự động clamp theo position (có heightReference trong billboard)
        return new Cesium.LabelGraphics({
            text: shortName,
            font: 'bold 13px "Segoe UI", Arial, sans-serif',
            fillColor: Cesium.Color.WHITE,
            outlineColor: fillColor,
            outlineWidth: 3,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            backgroundColor: backgroundColor,
            backgroundPadding: new Cesium.Cartesian2(8, 5),
            pixelOffset: new Cesium.Cartesian2(0, -32),
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 3e6, 0.3),
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 3e6),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            translucencyByDistance: new Cesium.NearFarScalar(5e4, 1.0, 3e6, 0.5),
        });
    }, [sourceTerminal, destinationTerminal]);

    const addTerminal = useCallback((terminal: UserTerminal) => {
        if (!viewerRef.current) return;
        const { terminalId, position } = terminal;
        const lon = Number(position?.longitude);
        const lat = Number(position?.latitude);
        if (isNaN(lon) || isNaN(lat)) return;

        // GIỐNG HỆT GROUND_STATION: dùng height 100 để gắn vào mặt đất
        const staticPos = Cesium.Cartesian3.fromDegrees(lon, lat, 100);
        const positionProperty = new Cesium.ConstantPositionProperty(staticPos);

        const existing = terminalEntityCacheRef.current.get(terminalId);
        if (existing) {
            viewerRef.current.entities.remove(existing);
        }

        // GIỐNG HỆT GROUND_STATION: chỉ có billboard + label (KHÔNG có point)
        const entity = viewerRef.current.entities.add({
            id: `TERMINAL-${terminalId}`,
            name: terminal.terminalName,
            position: positionProperty,
            billboard: createTerminalBillboard(terminal),
            label: createTerminalLabel(terminal),
        });
        terminalEntityCacheRef.current.set(terminalId, entity);
    }, [createTerminalBillboard, createTerminalLabel]);

    const batchAddTerminals = useCallback((terminalsToAdd: UserTerminal[]) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        viewer.entities.suspendEvents();
        terminalsToAdd.forEach(addTerminal);
        viewer.entities.resumeEvents();
    }, [addTerminal]);

    // ========== Helper: Calculate distance between two coordinates (Haversine) ==========
    const calculateDistanceKm = useCallback((lat1: number, lon1: number, lat2: number, lon2: number): number => {
        const R = 6371; // Earth's radius in kilometers
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = 
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }, []);

    // ========== Connection Lines ==========
    // Vẽ path màu vàng từ terminal đến ground station connected (chỉ trong phạm vi hợp lý)
    const updateConnectionLines = useCallback(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        // Remove old connection lines
        connectionLineCacheRef.current.forEach((line) => {
            viewer.entities.remove(line);
        });
        connectionLineCacheRef.current.clear();

        // Phạm vi mở rộng cho terminal-to-ground-station connection: 1500km
        // Để hỗ trợ kết nối xa (ví dụ: Đà Nẵng → Hà Nội ~750km, Đà Nẵng → Hồ Chí Minh ~900km)
        const MAX_CONNECTION_RANGE_KM = 1500; // Maximum range for terminal connection (mở rộng)

        // Vẽ path màu vàng cho mỗi terminal đã connect đến ground station
        let drawnCount = 0;
        let skippedCount = 0;
        let tooFarCount = 0;
        
        terminals.forEach((terminal) => {
            const connectedNodeId = terminal.connectedNodeId;
            
            // Chỉ cần có connectedNodeId là đủ (không cần kiểm tra status)
            if (!connectedNodeId) {
                skippedCount++;
                return;
            }

            // Tìm ground station node
            const connectedNode = nodes.find(n => n.nodeId === connectedNodeId);
            if (!connectedNode) {
                console.warn(`⚠️ Node not found for terminal ${terminal.terminalId}: ${connectedNodeId}`);
                skippedCount++;
                return;
            }
            
            // Kiểm tra node type - chỉ vẽ cho GROUND_STATION
            if (connectedNode.nodeType !== 'GROUND_STATION') {
                skippedCount++;
                return;
            }

            // Lấy vị trí terminal và ground station
            const terminalPos = terminal.position;
            const nodePos = connectedNode.position;

            if (!terminalPos || !nodePos) {
                console.warn(`⚠️ Missing position for terminal ${terminal.terminalId} or node ${connectedNodeId}`);
                skippedCount++;
                return;
            }

            // Tính khoảng cách giữa terminal và ground station
            const distanceKm = calculateDistanceKm(
                terminalPos.latitude,
                terminalPos.longitude,
                nodePos.latitude,
                nodePos.longitude
            );

            // Chỉ vẽ path nếu terminal trong phạm vi hợp lý (≤ 100km)
            if (distanceKm > MAX_CONNECTION_RANGE_KM) {
                console.warn(
                    `⚠️ Terminal ${terminal.terminalId} too far from GS ${connectedNodeId}: ` +
                    `${distanceKm.toFixed(1)}km (max: ${MAX_CONNECTION_RANGE_KM}km)`
                );
                tooFarCount++;
                return;
            }

            // Tạo positions cho polyline cong đẹp hơn với độ cao
            // Độ cao tối thiểu để không bị che bởi terrain (tăng theo khoảng cách)
            const minAltitude = Math.max(500, distanceKm * 10); // Ít nhất 500m, hoặc 10m/km
            const maxAltitude = minAltitude + 2000; // Điểm cao nhất ở giữa cao hơn 2km
            
            // Tính điểm giữa (midpoint) để tạo đường cong
            const midLat = (terminalPos.latitude + nodePos.latitude) / 2;
            const midLon = (terminalPos.longitude + nodePos.longitude) / 2;
            
            // Tạo 3 điểm để tạo đường cong đẹp hơn
            const positions = [
                Cesium.Cartesian3.fromDegrees(
                    terminalPos.longitude,
                    terminalPos.latitude,
                    minAltitude // Điểm bắt đầu có độ cao
                ),
                Cesium.Cartesian3.fromDegrees(
                    midLon,
                    midLat,
                    maxAltitude // Điểm giữa cao nhất để tạo đường cong
                ),
                Cesium.Cartesian3.fromDegrees(
                    nodePos.longitude,
                    nodePos.latitude,
                    minAltitude // Điểm kết thúc có độ cao
                )
            ];

            // Vẽ path màu vàng với glow effect đẹp hơn
            const connectionLineId = `TERMINAL-CONNECTION-${terminal.terminalId}`;
            const connectionLine = viewer.entities.add({
                id: connectionLineId,
                name: `Terminal Connection: ${terminal.terminalId} → ${connectedNodeId} (${distanceKm.toFixed(1)}km)`,
                polyline: {
                    positions: positions,
                    width: 3, // Tăng width để dễ nhìn hơn
                    material: new Cesium.PolylineGlowMaterialProperty({
                        glowPower: 0.3, // Glow effect
                        color: Cesium.Color.YELLOW.withAlpha(0.8), // Màu vàng với độ trong suốt
                    }),
                    clampToGround: false, // Không clamp để giữ độ cao
                    arcType: Cesium.ArcType.GEODESIC, // Đường cong theo geodesic
                    distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 1e7), // Hiển thị trong phạm vi
                },
            });

            connectionLineCacheRef.current.set(connectionLineId, connectionLine);
            drawnCount++;
        });
        
        console.log(
            `📊 Connection lines: ${drawnCount} drawn, ${tooFarCount} too far, ${skippedCount} skipped ` +
            `(total terminals: ${terminals.length})`
        );
    }, [terminals, nodes, calculateDistanceKm]);

    // ========== Draw Single Routing Path ==========
    const drawRoutingPath = useCallback((path: RoutingPath, pathId: string, isPacketPath: boolean = false) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        if (!path || path.path.length < 2) {
            console.warn('⚠️ Invalid path data:', path);
            return;
        }

        // Debug: Log path segments
        console.log(`📊 Drawing path ${pathId}:`, {
            algorithm: path.algorithm,
            hops: path.hops,
            segments: path.path.length,
            segments_detail: path.path.map(seg => ({
                type: seg.type,
                id: seg.id,
                name: seg.name,
                altitude: seg.position.altitude || 0
            }))
        });

        // Get color based on algorithm
        const pathColor = getAlgorithmColor(path.algorithm);
        const alpha = isPacketPath ? 0.6 : 0.8; // Slightly more transparent for packet paths

        // Convert path segments to Cartesian3 positions with smooth arc
        const positions: Cesium.Cartesian3[] = [];
        const originalPositions: Cesium.Cartesian3[] = [];
        
        // First, collect all original positions
        path.path.forEach((segment, index) => {
            const pos = Cesium.Cartesian3.fromDegrees(
                segment.position.longitude,
                segment.position.latitude,
                segment.position.altitude || 0
            );
            originalPositions.push(pos);
            
            // Debug: Log each segment position
            if (segment.type === 'node') {
                console.log(`  📍 Segment ${index}: ${segment.type} ${segment.name} at altitude ${segment.position.altitude || 0}m`);
            }
        });

        // Create smooth curved path with elevated arc
        for (let i = 0; i < originalPositions.length - 1; i++) {
            const startPos = originalPositions[i];
            const endPos = originalPositions[i + 1];
            
            // Calculate arc height based on distance (higher arc for longer segments)
            const distance = Cesium.Cartesian3.distance(startPos, endPos);
            const arcHeight = Math.max(100000, distance * 0.2); // Minimum 100km arc (increased from 50km), scales with distance
            
            // Interpolate positions along the arc
            const segments = Math.max(20, Math.floor(distance / 100000)); // More segments for smoother curves
            
            for (let j = 0; j <= segments; j++) {
                const t = j / segments;
                
                // Quadratic Bezier curve formula: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
                // We calculate the midpoint elevated by arcHeight as control point
                const midpoint = Cesium.Cartesian3.lerp(startPos, endPos, 0.5, new Cesium.Cartesian3());
                
                // Get surface normal at midpoint and elevate
                const midpointCartographic = Cesium.Cartographic.fromCartesian(midpoint);
                const elevatedMidpoint = Cesium.Cartesian3.fromRadians(
                    midpointCartographic.longitude,
                    midpointCartographic.latitude,
                    midpointCartographic.height + arcHeight
                );
                
                // Bezier interpolation
                const oneMinusT = 1 - t;
                const interpolated = new Cesium.Cartesian3();
                
                // (1-t)² * startPos
                const term1 = Cesium.Cartesian3.multiplyByScalar(startPos, oneMinusT * oneMinusT, new Cesium.Cartesian3());
                // 2(1-t)t * elevatedMidpoint
                const term2 = Cesium.Cartesian3.multiplyByScalar(elevatedMidpoint, 2 * oneMinusT * t, new Cesium.Cartesian3());
                // t² * endPos
                const term3 = Cesium.Cartesian3.multiplyByScalar(endPos, t * t, new Cesium.Cartesian3());
                
                Cesium.Cartesian3.add(term1, term2, interpolated);
                Cesium.Cartesian3.add(interpolated, term3, interpolated);
                
                positions.push(interpolated);
            }
        }
        
        // Add the final position
        positions.push(originalPositions[originalPositions.length - 1]);

        // Draw polyline for the entire path with smooth curve
        const pathLine = viewer.entities.add({
            id: pathId,
            name: `PATH-${pathId}`, // Add name for easier identification
            polyline: {
                positions: positions,
                width: isPacketPath ? 2 : 3, // Thinner lines for elegance
                material: pathColor.withAlpha(alpha),
                clampToGround: false,
                arcType: Cesium.ArcType.NONE, // Use NONE since we manually created the arc
                distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, Number.POSITIVE_INFINITY),
            },
        });
        
        // Store path data in separate cache for click handler
        pathDataCacheRef.current.set(pathId, path);

        if (isPacketPath) {
            // Store in packet path cache
            if (!packetPathCacheRef.current.has(pathId)) {
                packetPathCacheRef.current.set(pathId, []);
            }
            packetPathCacheRef.current.get(pathId)!.push(pathLine);
        } else {
            routingPathCacheRef.current.set(pathId, pathLine);
        }

        // Add markers for source and destination (only for single routing path, not packet paths)
        if (!isPacketPath) {
            const sourcePos = Cesium.Cartesian3.fromDegrees(
                path.source.position.longitude,
                path.source.position.latitude,
                path.source.position.altitude || 0
            );
            const destPos = Cesium.Cartesian3.fromDegrees(
                path.destination.position.longitude,
                path.destination.position.latitude,
                path.destination.position.altitude || 0
            );

            // Source marker (spring green - xanh lá neon)
            const sourceMarker = viewer.entities.add({
                id: `${pathId}-source`,
                position: sourcePos,
                point: {
                    pixelSize: 12,
                    color: Cesium.Color.SPRINGGREEN,
                    outlineColor: Cesium.Color.BLACK,
                    outlineWidth: 2,
                    heightReference: Cesium.HeightReference.NONE,
                },
                label: {
                    text: 'SRC',
                    font: 'bold 12px sans-serif',
                    fillColor: Cesium.Color.SPRINGGREEN,
                    outlineColor: Cesium.Color.BLACK,
                    outlineWidth: 2,
                    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                    pixelOffset: new Cesium.Cartesian2(0, -30),
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                },
            });
            routingPathCacheRef.current.set(`${pathId}-source`, sourceMarker);

            // Destination marker (orange red - đỏ cam)
            const destMarker = viewer.entities.add({
                id: `${pathId}-dest`,
                position: destPos,
                point: {
                    pixelSize: 12,
                    color: Cesium.Color.ORANGERED,
                    outlineColor: Cesium.Color.BLACK,
                    outlineWidth: 2,
                    heightReference: Cesium.HeightReference.NONE,
                },
                label: {
                    text: 'DEST',
                    font: 'bold 12px sans-serif',
                    fillColor: Cesium.Color.ORANGERED,
                    outlineColor: Cesium.Color.BLACK,
                    outlineWidth: 2,
                    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                    pixelOffset: new Cesium.Cartesian2(0, -30),
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                },
            });
            routingPathCacheRef.current.set(`${pathId}-dest`, destMarker);

            // Add intermediate node markers
            path.path.forEach((segment, index) => {
                if (segment.type === 'node' && index > 0 && index < path.path.length - 1) {
                    const nodePos = Cesium.Cartesian3.fromDegrees(
                        segment.position.longitude,
                        segment.position.latitude,
                        segment.position.altitude || 0
                    );
                    const nodeMarker = viewer.entities.add({
                        id: `${pathId}-node-${index}`,
                        position: nodePos,
                        point: {
                            pixelSize: 10,
                            color: pathColor,
                            outlineColor: Cesium.Color.BLACK,
                            outlineWidth: 2,
                            heightReference: Cesium.HeightReference.NONE,
                        },
                        label: {
                            text: segment.name,
                            font: '10px sans-serif',
                            fillColor: pathColor,
                            outlineColor: Cesium.Color.BLACK,
                            outlineWidth: 1,
                            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                            pixelOffset: new Cesium.Cartesian2(0, -25),
                            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                        },
                    });
                    routingPathCacheRef.current.set(`${pathId}-node-${index}`, nodeMarker);
                }
            });
        }
    }, []);

    // ========== Update Single Routing Path ==========
    const updateRoutingPath = useCallback(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        // Remove old routing path
        routingPathCacheRef.current.forEach((entity) => {
            viewer.entities.remove(entity);
        });
        routingPathCacheRef.current.clear();
        
        // Clear path data cache for routing paths
        pathDataCacheRef.current.forEach((_path, pathId) => {
            if (pathId.startsWith('ROUTING-')) {
                pathDataCacheRef.current.delete(pathId);
            }
        });

        if (!routingPath || routingPath.path.length < 2) {
            return;
        }

        const pathId = `ROUTING-${routingPath.source.terminalId}-${routingPath.destination.terminalId}`;
        drawRoutingPath(routingPath, pathId, false);
    }, [routingPath, drawRoutingPath]);

    // ========== Update Packet Paths ==========
    const updatePacketPaths = useCallback(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        // Remove old packet paths that are no longer in activePackets
        const activePacketIds = new Set(activePackets.map(p => p.packetId));
        packetPathCacheRef.current.forEach((entities, packetId) => {
            if (!activePacketIds.has(packetId)) {
                entities.forEach((entity) => {
                    viewer.entities.remove(entity);
                });
                packetPathCacheRef.current.delete(packetId);
                // Also remove from path data cache
                const pathId = `PACKET-PATH-${packetId}`;
                pathDataCacheRef.current.delete(pathId);
            }
        });

        // Draw paths for active packets
        activePackets.forEach((packet) => {
            if (packet.path && packet.path.path && packet.path.path.length >= 2) {
                const pathId = `PACKET-PATH-${packet.packetId}`;
                
                // Skip if path already drawn
                if (packetPathCacheRef.current.has(pathId)) {
                    return;
                }

                drawRoutingPath(packet.path, pathId, true);
            }
        });
    }, [activePackets, drawRoutingPath]);

    // ========== Clear All Paths ==========
    const clearAllPaths = useCallback(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;

        // Clear single routing path
        routingPathCacheRef.current.forEach((entity) => {
            viewer.entities.remove(entity);
        });
        routingPathCacheRef.current.clear();

        // Clear packet paths
        packetPathCacheRef.current.forEach((entities) => {
            entities.forEach((entity) => {
                viewer.entities.remove(entity);
            });
        });
        packetPathCacheRef.current.clear();
        
        // Clear all path data cache
        pathDataCacheRef.current.clear();

        if (onClearPaths) {
            onClearPaths();
        }
    }, [onClearPaths]);

    // Expose clearAllPaths to parent via window
    useEffect(() => {
        if (viewerRef.current) {
            (window.viewer as any).clearAllPaths = clearAllPaths;
        }
    }, [clearAllPaths]);

    // ========== Initialize Cesium (Chỉ chạy 1 lần) ==========
    useEffect(() => {
        if (cesiumContainer.current && !viewerRef.current) {
            const viewer = new Cesium.Viewer(cesiumContainer.current, {
                timeline: true,
                animation: true,
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
            
            // QUAN TRỌNG: Thêm terrain provider để CLAMP_TO_GROUND hoạt động đúng
            // Dùng EllipsoidTerrainProvider (miễn phí) hoặc Cesium World Terrain (cần token)
            // EllipsoidTerrainProvider sẽ clamp dựa trên ellipsoid (hình cầu), đủ cho mục đích này
            viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
            viewer.camera.setView({
                destination: Cesium.Cartesian3.fromDegrees(0, 0, 2e7),
            });
            viewer.clock.shouldAnimate = true;
            viewer.clock.multiplier = 1; // Normal speed for packet animations
            viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;

            viewerRef.current = viewer;
            window.viewer = viewer;
            
            return () => {
                viewer.destroy();
                viewerRef.current = null;
                window.viewer = undefined;
            };
        }
    }, []);

    // ========== useEffect cho Click Handler (Click map CHỈ HIỆN DETAIL) ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        
        let lastClickTime = 0;
        let lastClickPosition: Cesium.Cartesian2 | null = null;
        let singleClickTimeout: ReturnType<typeof setTimeout> | null = null;
        const DOUBLE_CLICK_DELAY = 300; // ms
        const DOUBLE_CLICK_DISTANCE_THRESHOLD = 5; // pixels
        
        const handleSingleClick = (event: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
            // Xử lý single-click
            const picked = viewer.scene.pick(event.position);
            
            console.log('🖱️ Click detected, picked:', picked);
            
            if (picked?.id) {
                const id = picked.id.id as string;
                console.log('🖱️ Clicked entity ID:', id);
                
                // Check if it's a routing path
                if ((id.startsWith('ROUTING-') || id.startsWith('PACKET-PATH-')) && onPathClick) {
                    const pathData = pathDataCacheRef.current.get(id);
                    if (pathData) {
                        console.log('🖱️ Clicked routing path:', id);
                        onPathClick(pathData);
                        return;
                    }
                }
                
                // Check if it's a terminal
                if (id.startsWith('TERMINAL-')) {
                    const terminalId = id.replace('TERMINAL-', '');
                    console.log('🖱️ Clicked terminal ID:', terminalId);
                    console.log('🖱️ Available terminals:', terminals.map(t => t.terminalId));
                    
                    const terminal = terminals.find(t => t.terminalId === terminalId);
                    
                    if (terminal) {
                        console.log('✅ Terminal found, setting selected:', terminal.terminalName);
                        
                        // Logic chọn source và destination
                        if (sourceTerminal?.terminalId === terminalId) {
                            setSourceTerminal(null);
                        } else if (destinationTerminal?.terminalId === terminalId) {
                            setDestinationTerminal(null);
                        } else if (!sourceTerminal) {
                            setSourceTerminal(terminal);
                        } else if (!destinationTerminal && sourceTerminal.terminalId !== terminalId) {
                            setDestinationTerminal(terminal);
                        }
                        
                        setSelectedTerminal(terminal);
                        setSelectedNode(null);
                    } else {
                        console.warn('⚠️ Terminal not found in store:', terminalId);
                    }
                } else {
                    // Check if it's a node
                    const node = nodes.find(n => n.nodeId === id);
                    if (node) {
                        console.log('🖱️ Clicked node:', node.nodeName);
                    }
                    setSelectedNode(node || null);
                    setSelectedTerminal(null);
                }
            } else {
                // Click vào map trống
                console.log('🖱️ Clicked empty map');
                setSelectedNode(null);
                setSelectedTerminal(null);
            }
        };
        
        const clickHandler = async (event: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
            const currentTime = Date.now();
            const currentPosition = event.position;
            
            // Kiểm tra double-click
            const isDoubleClick = lastClickTime && 
                currentTime - lastClickTime < DOUBLE_CLICK_DELAY &&
                lastClickPosition &&
                Cesium.Cartesian2.distance(lastClickPosition, currentPosition) < DOUBLE_CLICK_DISTANCE_THRESHOLD;
            
            if (isDoubleClick) {
                // Cancel single-click timeout nếu có
                if (singleClickTimeout) {
                    clearTimeout(singleClickTimeout);
                    singleClickTimeout = null;
                }
                
                // Double-click detected - kiểm tra xem có click vào entity không
                const picked = viewer.scene.pick(event.position);
                
                // Chỉ tạo terminal mới nếu double-click vào empty map (không phải entity)
                if (!picked || !picked.id) {
                    const cartesian = viewer.camera.pickEllipsoid(currentPosition, viewer.scene.globe.ellipsoid);
                    
                    if (cartesian) {
                        const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
                        const latitude = Cesium.Math.toDegrees(cartographic.latitude);
                        const longitude = Cesium.Math.toDegrees(cartographic.longitude);
                        const altitude = cartographic.height || 0;
                        
                        console.log(`📍 Double-click on empty map at (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`);
                        
                        try {
                            // Tạo terminal mới
                            const newTerminal = await createTerminalFromMap(
                                { latitude, longitude, altitude },
                                'MOBILE',
                                `Terminal at (${latitude.toFixed(2)}, ${longitude.toFixed(2)})`
                            );
                            
                            console.log('✅ Terminal created:', newTerminal);
                            
                            // Callback để update UI
                            if (onTerminalCreated) {
                                onTerminalCreated(newTerminal);
                            }
                            
                            // Refresh terminals từ store
                            const updatedTerminals = await getUserTerminals();
                            setTerminals(updatedTerminals);
                            
                        } catch (error) {
                            console.error('❌ Error creating terminal:', error);
                            alert(`Failed to create terminal: ${error instanceof Error ? error.message : 'Unknown error'}`);
                        }
                    }
                } else {
                    console.log('📍 Double-click on entity - ignoring terminal creation');
                }
                
                // Reset
                lastClickTime = 0;
                lastClickPosition = null;
                return;
            }
            
            // Single click - lưu thông tin để check double-click cho lần click tiếp theo
            lastClickTime = currentTime;
            lastClickPosition = currentPosition;
            
            // Clear timeout cũ nếu có
            if (singleClickTimeout) {
                clearTimeout(singleClickTimeout);
            }
            
            // Delay xử lý single-click để tránh conflict với double-click
            singleClickTimeout = setTimeout(() => {
                handleSingleClick(event);
                singleClickTimeout = null;
            }, DOUBLE_CLICK_DELAY);

        };

        viewer.screenSpaceEventHandler.setInputAction(
            clickHandler,
            Cesium.ScreenSpaceEventType.LEFT_CLICK
        );

        return () => {
            const currentViewer = viewerRef.current;
            if (currentViewer && !currentViewer.isDestroyed()) {
                currentViewer.screenSpaceEventHandler.removeInputAction(
                    Cesium.ScreenSpaceEventType.LEFT_CLICK
                );
            }
        };
    }, [nodes, terminals, setSelectedNode, setSelectedTerminal, sourceTerminal, destinationTerminal, setSourceTerminal, setDestinationTerminal, onPathClick, routingPath, activePackets, onTerminalCreated, setTerminals]); // Phụ thuộc vào `nodes`, `terminals`, và selection state

    // ========== Update Nodes & Clock ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        
        // Store terminal IDs before removing
        const terminalIds = Array.from(terminalEntityCacheRef.current.keys());
        
        // Remove only node entities, not terminals or connection lines
        entityCacheRef.current.forEach((entity) => {
            viewer.entities.remove(entity);
        });
        entityCacheRef.current.clear();
        
        if (nodes.length > 0) {
        batchAddNodes(nodes);
        }

        // Re-add terminals (cần update để reflect selection state)
        terminalIds.forEach(id => {
            const entity = terminalEntityCacheRef.current.get(id);
            if (entity) {
                viewer.entities.remove(entity);
            }
        });
        terminalEntityCacheRef.current.clear();
        // Re-add terminals với updated visual state
        if (terminals.length > 0) {
            batchAddTerminals(terminals);
        }
        
        // Update connection lines khi nodes thay đổi (đảm bảo paths được vẽ khi load map)
        updateConnectionLines();

        if (nodes.length > 0) {
        const startTime = Cesium.JulianDate.now();
        const stopTime = Cesium.JulianDate.addSeconds(startTime, ORBIT_CONFIG.LEO.period * 2, new Cesium.JulianDate());
        
        viewer.clock.startTime = startTime;
        viewer.clock.stopTime = stopTime;
        viewer.clock.currentTime = startTime;
        viewer.timeline?.zoomTo(startTime, stopTime);

        const firstSat = nodes.find(n => n.nodeType.includes("SATELLITE"));
        if (firstSat) {
            setTimeout(() => {
                if (!viewer || viewer.isDestroyed() || !viewer.entities) return;
                const entity = viewer.entities.getById(firstSat.nodeId);
                if (entity) {
                    viewer.flyTo(entity, { duration: 2 });
                }
            }, 500);
            }
        }
    }, [nodes, batchAddNodes, terminals, batchAddTerminals, updateConnectionLines]);

    // ========== Update Terminals ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        batchAddTerminals(terminals);
        updateConnectionLines();
    }, [terminals, nodes, batchAddTerminals, updateConnectionLines]);

    // ========== Update Routing Path ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        updateRoutingPath();
    }, [routingPath, updateRoutingPath]);

    // ========== Update Packet Paths ==========
    useEffect(() => {
        updatePacketPaths();
    }, [updatePacketPaths]);

    // ========== Packet Animations ==========
    // DISABLED: Only show paths, no packet animation for batch mode
    // useEffect(() => {
    //     if (!viewerRef.current) return;
    //     const viewer = viewerRef.current;

    //     console.log(`🎬 Updating packet animations, active packets: ${activePackets.length}`);

    //     // Remove old animations that are no longer in activePackets
    //     const activePacketIds = new Set(activePackets.map(p => p.packetId));
    //     packetAnimationsRef.current.forEach((animation, packetId) => {
    //         if (!activePacketIds.has(packetId)) {
    //             removePacketAnimation(viewer, animation);
    //             packetAnimationsRef.current.delete(packetId);
    //         }
    //     });

    //     // Create new animations for active packets that don't have animations yet
    //     activePackets.forEach((packet) => {
    //         if (packet.path && packet.path.path && packet.path.path.length >= 2) {
    //             // Skip if animation already exists
    //             if (packetAnimationsRef.current.has(packet.packetId)) {
    //                 return;
    //             }
                
    //             // Calculate animation duration based on estimated latency (slow motion)
    //             // Make it slower for better visibility: estimatedLatency / 20 instead of / 50
    //             const duration = Math.max(8, packet.path.estimatedLatency / 20); // Convert ms to seconds, min 8s for visibility
    //             const animation = createPacketAnimation(viewer, packet.packetId, packet.path, duration);
    //             if (animation) {
    //                 packetAnimationsRef.current.set(packet.packetId, animation);
    //                 processedPacketIdsRef.current.add(packet.packetId);
    //                 console.log(`✅ Created animation for packet ${packet.packetId}, duration: ${duration}s, path: ${packet.path.path.length} segments`);
                    
    //                 // Ensure clock is animating
    //                 if (!viewer.clock.shouldAnimate) {
    //                     viewer.clock.shouldAnimate = true;
    //                 }
                    
    //                 // Remove animation after it completes
    //                 setTimeout(() => {
    //                     if (packetAnimationsRef.current.has(packet.packetId)) {
    //                         removePacketAnimation(viewer, animation);
    //                         packetAnimationsRef.current.delete(packet.packetId);
    //                         processedPacketIdsRef.current.delete(packet.packetId);
    //                         console.log(`🗑️ Removed animation for packet ${packet.packetId}`);
    //                     }
    //                 }, duration * 1000 + 5000); // Add 5s buffer
    //             } else {
    //                 console.warn(`⚠️ Failed to create animation for packet ${packet.packetId}`);
    //             }
    //         } else {
    //             console.warn(`⚠️ Packet ${packet.packetId} has invalid path:`, packet.path);
    //         }
    //     });
    // }, [activePackets]);

    // ========== Camera Follow ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        let followListener: Cesium.Event.RemoveCallback | undefined;

        if (selectedNode && cameraFollowMode) {
            const entity = viewer.entities.getById(selectedNode.nodeId);
            if (!entity) return;

            const follow = (clock: Cesium.Clock) => {
                const pos = entity.position?.getValue(clock.currentTime);
                if (pos) {
                    viewer.camera.lookAt(pos, new Cesium.HeadingPitchRange(0, -Cesium.Math.PI_OVER_THREE, 2e5));
                }
            };
            followListener = viewer.clock.onTick.addEventListener(follow);
        }

        return () => {
            if (followListener) {
                followListener();
            }
        };
    }, [selectedNode, cameraFollowMode]);

    // Effect riêng để reset camera khi TẮT follow mode
    useEffect(() => {
        if (viewerRef.current && !cameraFollowMode) {
            viewerRef.current.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
        }
    }, [cameraFollowMode]);

    // ========== [MỚI] useEffect để XỬ LÝ BAY (từ Sidebar) ==========
    useEffect(() => {
        // Chỉ chạy nếu trigger THỰC SỰ thay đổi
        if (flyToTrigger === lastTriggerRef.current) {
            return;
        }

        // Cập nhật ref
        lastTriggerRef.current = flyToTrigger;
        
        if (!viewerRef.current || !selectedNode) {
            return;
        }

        const viewer = viewerRef.current;
        const entity = viewer.entities.getById(selectedNode.nodeId);
        
        if (entity) {
            const isSatellite = selectedNode.nodeType.includes("SATELLITE");
            
            viewer.flyTo(entity, {
                duration: 1.5,
                offset: new Cesium.HeadingPitchRange(
                    0,
                    -Cesium.Math.PI_OVER_THREE,
                    isSatellite ? 800000 : 100000 
                ),
            }).then((finished) => {
    if (finished) {
        setCameraFollowMode(true);
    }
});
        }
    }, [flyToTrigger, selectedNode, cameraFollowMode,setCameraFollowMode ]); 

    return <div ref={cesiumContainer} className="w-full h-full" style={{ pointerEvents: 'auto' }} />;
};

export default CesiumViewer;
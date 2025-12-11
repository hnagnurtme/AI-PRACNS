import React, { useRef, useState, useMemo } from 'react';
import Draggable from 'react-draggable';
import * as Cesium from 'cesium';
import { useTerminalStore } from '../../state/terminalStore';
import { useNodeStore } from '../../state/nodeStore';
import { connectTerminalToNode, disconnectTerminal } from '../../services/userTerminalService';
import { useNodes } from '../../hooks/useNodes';
import type { UserTerminal } from '../../types/UserTerminalTypes';

interface TerminalDetailCardProps {
    terminal: UserTerminal;
}

const DetailItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div>
        <dt className="text-gray-500 text-xs">{label}</dt>
        <dd className="font-medium text-gray-900 text-sm">{value}</dd>
    </div>
);

// Helper: Calculate distance between two coordinates (Haversine)
const calculateDistanceKm = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371; // Earth's radius in kilometers
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
};

const TerminalDetailCard: React.FC<TerminalDetailCardProps> = ({ terminal }) => {
    const { setSelectedTerminal, updateTerminalInStore, terminals } = useTerminalStore();
    const { nodes } = useNodeStore();
    const { refetchNodes } = useNodes();
    const terminalRef = useRef(null);
    const [isConnecting, setIsConnecting] = useState(false);
    const [isDisconnecting, setIsDisconnecting] = useState(false);
    const [selectedNodeId, setSelectedNodeId] = useState<string>('');
    const [showNodeSelector, setShowNodeSelector] = useState(false);

    const connectedNode = terminal.connectedNodeId
        ? nodes.find((n) => n.nodeId === terminal.connectedNodeId)
        : null;

    // Get available Ground Stations within range (mở rộng lên 1500km)
    // Để hỗ trợ kết nối xa (ví dụ: Đà Nẵng → Hà Nội ~750km, Đà Nẵng → Hồ Chí Minh ~900km)
    const availableGroundStations = useMemo(() => {
        const terminalPos = terminal.position;
        const MAX_RANGE_KM = 1500;
        
        return nodes
            .filter(node => node.nodeType === 'GROUND_STATION' && node.isOperational)
            .map(node => {
                const distance = calculateDistanceKm(
                    terminalPos.latitude,
                    terminalPos.longitude,
                    node.position.latitude,
                    node.position.longitude
                );
                return { node, distance };
            })
            .filter(({ distance }) => distance <= MAX_RANGE_KM)
            .sort((a, b) => a.distance - b.distance)
            .map(({ node, distance }) => ({
                ...node,
                distanceKm: distance
            }));
    }, [nodes, terminal.position]);

    // Count terminals connected to a GS
    const getTerminalCountForGS = (nodeId: string): number => {
        return terminals.filter(t => 
            t.connectedNodeId === nodeId && 
            (t.status === 'connected' || t.status === 'transmitting')
        ).length;
    };

    const handleConnect = async (nodeId?: string) => {
        const targetNodeId = nodeId || selectedNodeId;
        if (!targetNodeId) {
            alert('Please select a Ground Station to connect to');
            return;
        }

            setIsConnecting(true);
            try {
                const result = await connectTerminalToNode({
                    terminalId: terminal.terminalId,
                nodeId: targetNodeId,
                });
                if (result.success) {
                    updateTerminalInStore({
                        ...terminal,
                        status: 'connected',
                    connectedNodeId: targetNodeId,
                        connectionMetrics: {
                            latencyMs: result.latencyMs,
                            bandwidthMbps: result.bandwidthMbps,
                            packetLossRate: result.packetLossRate,
                        },
                        lastUpdated: result.timestamp,
                    });
                
                // Refresh nodes to get updated resource utilization
                await refetchNodes();
                
                setShowNodeSelector(false);
                setSelectedNodeId('');
                }
            } catch (error) {
                console.error('Failed to connect terminal:', error);
            alert(`Failed to connect terminal: ${error instanceof Error ? error.message : 'Unknown error'}`);
            } finally {
                setIsConnecting(false);
        }
    };

    const handleDisconnect = async () => {
        setIsDisconnecting(true);
        try {
            await disconnectTerminal(terminal.terminalId);
            updateTerminalInStore({
                ...terminal,
                status: 'idle',
                connectedNodeId: null,
                connectionMetrics: undefined,
                lastUpdated: new Date().toISOString(),
            });
            
            // Refresh nodes to get updated resource utilization
            await refetchNodes();
        } catch (error) {
            console.error('Failed to disconnect terminal:', error);
            alert('Failed to disconnect terminal. Please try again.');
        } finally {
            setIsDisconnecting(false);
        }
    };

    const handleFlyTo = () => {
        // Fly to terminal on map
        if (window.viewer) {
            const entity = window.viewer.entities.getById(`TERMINAL-${terminal.terminalId}`);
            if (entity) {
                window.viewer.flyTo(entity, {
                    duration: 1.5,
                    offset: new Cesium.HeadingPitchRange(0, -Cesium.Math.PI_OVER_THREE, 50000),
                });
            }
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'idle':
                return 'bg-gray-100 text-gray-700';
            case 'connected':
                return 'bg-green-100 text-green-700';
            case 'transmitting':
                return 'bg-yellow-100 text-yellow-700';
            case 'disconnected':
                return 'bg-red-100 text-red-700';
            default:
                return 'bg-gray-100 text-gray-700';
        }
    };

    const getTerminalTypeColor = (type: string) => {
        switch (type) {
            case 'MOBILE':
                return 'bg-blue-100 text-blue-700';
            case 'FIXED':
                return 'bg-purple-100 text-purple-700';
            case 'VEHICLE':
                return 'bg-orange-100 text-orange-700';
            case 'AIRCRAFT':
                return 'bg-indigo-100 text-indigo-700';
            default:
                return 'bg-gray-100 text-gray-700';
        }
    };

    return (
        <Draggable 
            handle=".drag-handle" 
            bounds="parent" 
            nodeRef={terminalRef}
            cancel="button, input, select, textarea, .no-drag, .clickable"
            enableUserSelectHack={false}
        >
            <div
                ref={terminalRef}
                className="absolute top-20 right-4 z-50 w-[450px] 
                           bg-white rounded-xl shadow-xl border border-gray-200
                           flex flex-col"
                onPointerDown={(e) => {
                    // Allow clicks on buttons to work
                    if ((e.target as HTMLElement).closest('button, .clickable')) {
                        e.stopPropagation();
                    }
                }}
            >
                {/* Header */}
                <div className="drag-handle flex justify-between items-start p-4 pb-2 cursor-move">
                    <div className="flex-1 truncate">
                        <h3 className="text-xl font-extrabold text-gray-800 truncate select-none">
                            {terminal.terminalName}
                        </h3>
                        <p className="text-xs text-gray-500 select-none">
                            ID: {terminal.terminalId.substring(0, 8)}...
                        </p>
                    </div>
                    <button
                        onClick={() => setSelectedTerminal(null)}
                        className="text-gray-500 hover:text-red-600 text-lg font-bold ml-2"
                    >
                        &times;
                    </button>
                </div>

                {/* Status Badges */}
                <div className="px-4 pb-3 border-b border-gray-200">
                    <p className="text-sm flex flex-wrap gap-2 items-center">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${getStatusColor(terminal.status)}`}>
                            {terminal.status.toUpperCase()}
                        </span>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${getTerminalTypeColor(terminal.terminalType)}`}>
                            {terminal.terminalType}
                        </span>
                    </p>
                </div>

                {/* Content */}
                <div className="p-4 bg-gray-50 max-h-96 overflow-y-auto">
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">📍 Position</h4>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-4">
                        <DetailItem label="Latitude" value={terminal.position.latitude.toFixed(4)} />
                        <DetailItem label="Longitude" value={terminal.position.longitude.toFixed(4)} />
                        <DetailItem label="Altitude" value={`${terminal.position.altitude.toFixed(2)} m`} />
                    </div>

                    {terminal.connectedNodeId && connectedNode && (
                        <>
                            <h4 className="text-sm font-semibold text-gray-700 mb-2 mt-4">🔗 Connected Ground Station</h4>
                            <div className="mb-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
                                <div className="flex justify-between items-start mb-2">
                                    <div>
                                        <p className="text-sm font-semibold text-gray-900">{connectedNode.nodeName}</p>
                                        <p className="text-xs text-gray-500">ID: {connectedNode.nodeId.substring(0, 12)}...</p>
                                    </div>
                                    <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded">
                                        CONNECTED
                                    </span>
                                </div>
                                
                                {/* Resource Utilization của GS */}
                                <div className="mt-3 pt-3 border-t border-blue-200">
                                    <h5 className="text-xs font-semibold text-gray-700 mb-2">📊 Resource Status</h5>
                                    <div className="space-y-2">
                                        <div>
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-xs text-gray-600">Utilization</span>
                                                <span className={`text-xs font-bold ${
                                                    (connectedNode.resourceUtilization || 0) > 80 ? 'text-red-600' :
                                                    (connectedNode.resourceUtilization || 0) > 60 ? 'text-yellow-600' :
                                                    'text-green-600'
                                                }`}>
                                                    {(connectedNode.resourceUtilization || 0).toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="w-full bg-gray-200 rounded-full h-2">
                                                <div 
                                                    className={`h-2 rounded-full transition-all ${
                                                        (connectedNode.resourceUtilization || 0) > 80 ? 'bg-red-500' :
                                                        (connectedNode.resourceUtilization || 0) > 60 ? 'bg-yellow-500' :
                                                        'bg-green-500'
                                                    }`}
                                                    style={{ width: `${Math.min((connectedNode.resourceUtilization || 0), 100)}%` }}
                                                />
                                            </div>
                                        </div>
                                        
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            <div>
                                                <span className="text-gray-600">Connected Terminals:</span>
                                                <span className="font-bold text-gray-900 ml-1">
                                                    {getTerminalCountForGS(connectedNode.nodeId)}
                                                </span>
                                            </div>
                                            <div>
                                                <span className="text-gray-600">Battery:</span>
                                                <span className={`font-bold ml-1 ${
                                                    (connectedNode.batteryChargePercent || 100) < 30 ? 'text-red-600' :
                                                    (connectedNode.batteryChargePercent || 100) < 50 ? 'text-yellow-600' :
                                                    'text-green-600'
                                                }`}>
                                                    {(connectedNode.batteryChargePercent || 100).toFixed(0)}%
                                                </span>
                                            </div>
                                            <div>
                                                <span className="text-gray-600">Packet Loss:</span>
                                                <span className={`font-bold ml-1 ${
                                                    (connectedNode.packetLossRate || 0) > 0.05 ? 'text-red-600' :
                                                    (connectedNode.packetLossRate || 0) > 0.01 ? 'text-yellow-600' :
                                                    'text-green-600'
                                                }`}>
                                                    {((connectedNode.packetLossRate || 0) * 100).toFixed(2)}%
                                                </span>
                                            </div>
                                            <div>
                                                <span className="text-gray-600">Queue:</span>
                                                <span className={`font-bold ml-1 ${
                                                    ((connectedNode.currentPacketCount || 0) / (connectedNode.packetBufferCapacity || 1)) > 0.8 ? 'text-red-600' :
                                                    ((connectedNode.currentPacketCount || 0) / (connectedNode.packetBufferCapacity || 1)) > 0.5 ? 'text-yellow-600' :
                                                    'text-green-600'
                                                }`}>
                                                    {connectedNode.currentPacketCount || 0}/{connectedNode.packetBufferCapacity || 0}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}

                    {terminal.connectionMetrics && (
                        <>
                            <h4 className="text-sm font-semibold text-gray-700 mb-2 mt-4">📊 Connection Metrics</h4>
                            <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-4">
                                {terminal.connectionMetrics.latencyMs !== undefined && (
                                    <DetailItem label="Latency" value={`${terminal.connectionMetrics.latencyMs.toFixed(2)} ms`} />
                                )}
                                {terminal.connectionMetrics.bandwidthMbps !== undefined && (
                                    <DetailItem label="Bandwidth" value={`${terminal.connectionMetrics.bandwidthMbps.toFixed(2)} Mbps`} />
                                )}
                                {terminal.connectionMetrics.packetLossRate !== undefined && (
                                    <DetailItem label="Packet Loss" value={`${(terminal.connectionMetrics.packetLossRate * 100).toFixed(3)}%`} />
                                )}
                                {terminal.connectionMetrics.signalStrength !== undefined && (
                                    <DetailItem label="Signal Strength" value={`${terminal.connectionMetrics.signalStrength.toFixed(1)} dB`} />
                                )}
                            </div>
                        </>
                    )}

                    <h4 className="text-sm font-semibold text-gray-700 mb-2 mt-4">⚙️ QoS Requirements</h4>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-4">
                        <DetailItem label="Max Latency" value={`${terminal.qosRequirements.maxLatencyMs} ms`} />
                        <DetailItem label="Min Bandwidth" value={`${terminal.qosRequirements.minBandwidthMbps} Mbps`} />
                        <DetailItem label="Max Loss Rate" value={`${(terminal.qosRequirements.maxLossRate * 100).toFixed(3)}%`} />
                        <DetailItem label="Priority" value={`${terminal.qosRequirements.priority}`} />
                        {terminal.qosRequirements.serviceType && (
                            <DetailItem label="Service Type" value={terminal.qosRequirements.serviceType} />
                        )}
                    </div>

                    {terminal.metadata && (
                        <>
                            <h4 className="text-sm font-semibold text-gray-700 mb-2 mt-4">ℹ️ Metadata</h4>
                            <div className="text-xs text-gray-600">
                                {terminal.metadata.description && <p>{terminal.metadata.description}</p>}
                                {terminal.metadata.region && <p>Region: {terminal.metadata.region}</p>}
                            </div>
                        </>
                    )}

                    {terminal.lastUpdated && (
                        <div className="mt-4 pt-3 border-t border-gray-100">
                            <h4 className="text-sm font-semibold text-gray-700 mb-1">Last Updated</h4>
                            <div className="text-xs text-gray-500">
                                {new Date(terminal.lastUpdated).toLocaleString()}
                            </div>
                        </div>
                    )}
                </div>

                {/* Ground Station Selection (when not connected) */}
                {!terminal.connectedNodeId && showNodeSelector && (
                    <div className="px-4 pb-3 border-t border-gray-200 bg-blue-50 no-drag">
                        <h4 className="text-sm font-semibold text-gray-700 mb-2 mt-3">📡 Select Ground Station</h4>
                        {availableGroundStations.length === 0 ? (
                            <p className="text-xs text-red-600 mb-2">
                                ⚠️ No Ground Stations available within 100km range
                            </p>
                        ) : (
                            <div className="space-y-2 max-h-48 overflow-y-auto">
                                {availableGroundStations.map((gs) => {
                                    const terminalCount = getTerminalCountForGS(gs.nodeId);
                                    const utilization = gs.resourceUtilization || 0;
                                    return (
                                        <button
                                            key={gs.nodeId}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                e.preventDefault();
                                                handleConnect(gs.nodeId);
                                            }}
                                            onMouseDown={(e) => {
                                                e.stopPropagation();
                                                e.preventDefault();
                                            }}
                                            onPointerDown={(e) => {
                                                e.stopPropagation();
                                            }}
                                            disabled={isConnecting}
                                            className={`clickable w-full text-left p-2 rounded border transition-all ${
                                                selectedNodeId === gs.nodeId
                                                    ? 'border-blue-500 bg-blue-100'
                                                    : 'border-gray-200 bg-white hover:border-blue-300 hover:bg-blue-50'
                                            } ${isConnecting ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                                        >
                                            <div className="flex justify-between items-start">
                                                <div className="flex-1">
                                                    <p className="text-sm font-semibold text-gray-900">{gs.nodeName}</p>
                                                    <p className="text-xs text-gray-500">{gs.distanceKm.toFixed(1)} km away</p>
                                                </div>
                                                <div className="text-right">
                                                    <div className={`text-xs font-bold ${
                                                        utilization > 80 ? 'text-red-600' :
                                                        utilization > 60 ? 'text-yellow-600' :
                                                        'text-green-600'
                                                    }`}>
                                                        {utilization.toFixed(0)}%
                                                    </div>
                                                    <div className="text-xs text-gray-500">
                                                        {terminalCount} terminals
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="mt-1 w-full bg-gray-200 rounded-full h-1.5">
                                                <div 
                                                    className={`h-1.5 rounded-full ${
                                                        utilization > 80 ? 'bg-red-500' :
                                                        utilization > 60 ? 'bg-yellow-500' :
                                                        'bg-green-500'
                                                    }`}
                                                    style={{ width: `${Math.min(utilization, 100)}%` }}
                                                />
                                            </div>
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                setShowNodeSelector(false);
                                setSelectedNodeId('');
                            }}
                            onMouseDown={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                            }}
                            onPointerDown={(e) => {
                                e.stopPropagation();
                            }}
                            className="clickable mt-2 text-xs text-gray-600 hover:text-gray-800"
                        >
                            Cancel
                        </button>
                    </div>
                )}

                {/* Footer Actions */}
                <div className="p-4 flex flex-wrap gap-2 border-t border-gray-200 bg-white rounded-b-xl no-drag">
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            handleFlyTo();
                        }}
                        onMouseDown={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                        }}
                        onPointerDown={(e) => {
                            e.stopPropagation();
                        }}
                        className="clickable text-sm font-medium px-3 py-1 rounded border border-blue-200 text-blue-600 hover:bg-blue-50 transition-colors"
                    >
                        🗺️ Fly To
                    </button>
                    {terminal.status === 'connected' ? (
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                handleDisconnect();
                            }}
                            onMouseDown={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                            }}
                            onPointerDown={(e) => {
                                e.stopPropagation();
                            }}
                            disabled={isDisconnecting}
                            className={`clickable text-sm font-medium px-3 py-1 rounded border transition-colors ${
                                isDisconnecting
                                    ? 'bg-gray-400 text-white border-gray-400 cursor-not-allowed'
                                    : 'bg-red-600 text-white border-red-600 hover:bg-red-700'
                            }`}
                        >
                            {isDisconnecting ? 'Disconnecting...' : '🔌 Disconnect'}
                        </button>
                    ) : (
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                if (availableGroundStations.length === 0) {
                                    alert('No Ground Stations available within range. Please check terminal position.');
                                    return;
                                }
                                setShowNodeSelector(!showNodeSelector);
                            }}
                            onMouseDown={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                            }}
                            onPointerDown={(e) => {
                                e.stopPropagation();
                            }}
                            disabled={isConnecting || availableGroundStations.length === 0}
                            className={`clickable text-sm font-medium px-3 py-1 rounded border transition-colors ${
                                isConnecting || availableGroundStations.length === 0
                                    ? 'bg-gray-400 text-white border-gray-400 cursor-not-allowed'
                                    : 'bg-green-600 text-white border-green-600 hover:bg-green-700'
                            }`}
                        >
                            {isConnecting ? 'Connecting...' : '🔗 Connect to GS'}
                        </button>
                    )}
                </div>
            </div>
        </Draggable>
    );
};

export default TerminalDetailCard;


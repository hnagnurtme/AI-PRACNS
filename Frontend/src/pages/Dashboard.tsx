// src/Dashboard.tsx
import React, { useEffect, useState } from 'react';
import CesiumViewer from '../map/CesiumViewer';
import Sidebar from '../components/Sidebar';
import NodeDetailCard from '../components/nodes/NodeDetailCard';
import TerminalDetailCard from '../components/terminals/TerminalDetailCard';
import PacketSender from '../components/routing/PacketSender';
import SendPacketCard from '../components/routing/SendPacketCard';
import PathDetailCard from '../components/routing/PathDetailCard';
import { useNodeStore } from '../state/nodeStore';
import { useTerminalStore } from '../state/terminalStore';
import { useNodes } from '../hooks/useNodes';
import { getUserTerminals } from '../services/userTerminalService';
import { useUserTerminals } from '../hooks/useUserTerminals';
import { useWebSocket } from '../contexts/WebSocketContext';
import type { TerminalUpdate, TerminalConnectionResult } from '../types/UserTerminalTypes';
import type { RoutingPath, Packet } from '../types/RoutingTypes';

const Dashboard: React.FC = () => {
    const { nodes, selectedNode } = useNodeStore();
    const { terminals, selectedTerminal, sourceTerminal, destinationTerminal, updateTerminalInStore } = useTerminalStore();
    const { refetchNodes } = useNodes();
    const { 
        loading: terminalsLoading, 
        error: terminalsError,
        clearAllTerminals 
    } = useUserTerminals();
    const { subscribeToTerminalUpdates, subscribeToTerminalConnections } = useWebSocket();

    const [showRightSidebar, setShowRightSidebar] = useState(true);
    const [showPacketSender, setShowPacketSender] = useState(false);
    const [currentRoutingPath, setCurrentRoutingPath] = useState<RoutingPath | null>(null);
    const [selectedPath, setSelectedPath] = useState<RoutingPath | null>(null);
    const [activePackets, setActivePackets] = useState<Packet[]>([]);

    useEffect(() => {
        refetchNodes().catch((error) =>
            console.error('Failed to load Nodes data from API:', error)
        );
    }, [refetchNodes]);

    // Subscribe to terminal updates via WebSocket
    useEffect(() => {
        const unsubscribeUpdates = subscribeToTerminalUpdates((update: TerminalUpdate) => {
            updateTerminalInStore({
                ...terminals.find(t => t.terminalId === update.terminalId)!,
                ...update,
            });
        });

        const unsubscribeConnections = subscribeToTerminalConnections((result: TerminalConnectionResult) => {
            const terminal = terminals.find(t => t.terminalId === result.terminalId);
            if (terminal) {
                updateTerminalInStore({
                    ...terminal,
                    status: result.success ? 'connected' : 'idle',
                    connectedNodeId: result.success ? result.nodeId : null,
                    connectionMetrics: result.success ? {
                        latencyMs: result.latencyMs,
                        bandwidthMbps: result.bandwidthMbps,
                        packetLossRate: result.packetLossRate,
                    } : undefined,
                    lastUpdated: result.timestamp,
                });
            }
        });

        return () => {
            unsubscribeUpdates();
            unsubscribeConnections();
        };
    }, [terminals, subscribeToTerminalUpdates, subscribeToTerminalConnections, updateTerminalInStore]);

    // Removed: handleGenerateTerminals - no longer needed

    const handleClearTerminals = async () => {
        if (window.confirm('Are you sure you want to clear all terminals?')) {
            try {
                await clearAllTerminals();
            } catch (error) {
                console.error('Failed to clear terminals:', error);
                alert('Failed to clear terminals. Please try again.');
            }
        }
    };

    return (
        <div className="flex h-screen w-screen overflow-hidden bg-gray-50">
            {/* Left Sidebar - Nodes */}
            <div className="h-full flex-shrink-0 overflow-y-auto bg-white shadow-lg">
                <Sidebar nodes={nodes} />
            </div>

            {/* Map Area */}
            <div className="relative flex-grow h-full overflow-hidden">
                <CesiumViewer 
                    nodes={nodes} 
                    routingPath={currentRoutingPath} 
                    activePackets={activePackets}
                    onClearPaths={() => {
                        setCurrentRoutingPath(null);
                        setActivePackets([]);
                        setSelectedPath(null);
                    }}
                    onPathClick={(path: RoutingPath) => {
                        setSelectedPath(path);
                    }}
                    onTerminalCreated={async (terminal) => {
                        console.log('✅ New terminal created:', terminal);
                        // Refresh terminals list
                        const updatedTerminals = await getUserTerminals();
                        const { setTerminals } = useTerminalStore.getState();
                        setTerminals(updatedTerminals);
                    }}
                />

                {/* Send Packet Card - Hiển thị khi đã chọn 2 terminals */}
                {sourceTerminal && destinationTerminal && (
                    <SendPacketCard
                        onPacketSent={(packet: Packet) => {
                            setCurrentRoutingPath(packet.path);
                            setActivePackets((prev) => [...prev, packet]);
                        }}
                        onPathCalculated={(path: RoutingPath) => {
                            setCurrentRoutingPath(path);
                        }}
                    />
                )}

                {/* Path Detail Card - Hiển thị khi click vào path */}
                {selectedPath && (
                    <PathDetailCard
                        path={selectedPath}
                        nodes={nodes}
                        onClose={() => setSelectedPath(null)}
                    />
                )}

                {/* Packet Sender Panel */}
                {showPacketSender && (
                    <div className="absolute top-4 left-4 z-10">
                        <PacketSender
                            onPathCalculated={(path: RoutingPath) => {
                                setCurrentRoutingPath(path);
                            }}
                            onPacketSent={(packet: Packet) => {
                                setCurrentRoutingPath(packet.path);
                                setActivePackets((prev) => [...prev, packet]);
                            }}
                        />
                    </div>
                )}

                {/* Node Detail Card */}
                {selectedNode && (
                    <NodeDetailCard node={selectedNode} />
                )}

                {/* Terminal Detail Card */}
                {selectedTerminal && (
                    <TerminalDetailCard terminal={selectedTerminal} />
                )}

                {/* Toggle Button - Show Sidebar */}
                {!showRightSidebar && (
                    <button
                        onClick={() => setShowRightSidebar(true)}
                        className="absolute top-4 right-4 z-50 bg-slate-800 hover:bg-slate-700 text-white p-2.5 rounded-lg shadow-2xl transition-all hover:scale-105"
                        title="Show sidebar"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                        </svg>
                    </button>
                )}
            </div>

            {/* Right Sidebar - Controls */}
            {showRightSidebar && (
                <div className="h-full w-80 flex-shrink-0 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 shadow-2xl overflow-y-auto border-l border-slate-700">
                    <div className="p-4 space-y-4">
                        {/* Header */}
                        <div className="flex items-center justify-between pb-3 border-b border-slate-700">
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                                    </svg>
                                </div>
                                <div>
                                    <h2 className="text-sm font-bold text-white">Control Panel</h2>
                                    <p className="text-[10px] text-slate-400">Network Management</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setShowRightSidebar(false)}
                                className="text-slate-400 hover:text-white hover:bg-slate-700 p-1.5 rounded transition-all"
                                title="Hide sidebar"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Quick Actions */}
                        <div className="space-y-2">
                            <h3 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                                <svg className="w-3.5 h-3.5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                                Quick Actions
                            </h3>
                            <button
                                onClick={() => setShowPacketSender(!showPacketSender)}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-lg font-medium text-sm transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                                </svg>
                                Send Packet
                            </button>

                            {(currentRoutingPath || activePackets.length > 0) && (
                                <button
                                    onClick={() => {
                                        setCurrentRoutingPath(null);
                                        setActivePackets([]);
                                        if (window.viewer && (window.viewer as any).clearAllPaths) {
                                            (window.viewer as any).clearAllPaths();
                                        }
                                    }}
                                    className="w-full bg-slate-700/80 hover:bg-slate-600 text-white px-4 py-2.5 rounded-lg font-medium text-sm transition-all flex items-center justify-between group border border-slate-600"
                                >
                                    <div className="flex items-center gap-2">
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                        <span>Clear All Paths</span>
                                    </div>
                                    {activePackets.length > 0 && (
                                        <span className="bg-red-500 text-white rounded-full px-2 py-0.5 text-xs font-bold shadow-lg">
                                            {activePackets.length}
                                        </span>
                                    )}
                                </button>
                            )}
                        </div>

                        {/* Active Routing Info */}
                        {currentRoutingPath && (
                            <div className="bg-gradient-to-br from-indigo-900/50 to-purple-900/50 rounded-lg p-3 border border-indigo-700/50">
                                <h3 className="text-xs font-semibold text-indigo-300 mb-2 flex items-center gap-1.5">
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                    </svg>
                                    Active Route
                                </h3>
                                <div className="space-y-1.5 text-xs">
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Algorithm:</span>
                                        <span className="text-white font-semibold">{currentRoutingPath.algorithm || 'N/A'}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Hops:</span>
                                        <span className="text-white font-semibold">{currentRoutingPath.path?.length || 0}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Distance:</span>
                                        <span className="text-white font-semibold">{(currentRoutingPath.totalDistance || 0).toFixed(0)} km</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Latency:</span>
                                        <span className="text-white font-semibold">{(currentRoutingPath.estimatedLatency || 0).toFixed(2)} ms</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Terminals Section */}
                        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                                    <svg className="w-3.5 h-3.5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                    </svg>
                                    User Terminals
                                    <span className="ml-1 px-1.5 py-0.5 bg-purple-500/20 text-purple-300 rounded text-[10px] font-bold">{terminals.length}</span>
                                </h3>
                                {terminals.length > 0 && (
                                    <button
                                        onClick={handleClearTerminals}
                                        disabled={terminalsLoading}
                                        className="text-[10px] text-red-400 hover:text-red-300 hover:bg-red-500/10 px-2 py-1 rounded font-medium disabled:text-slate-600 transition-all"
                                    >
                                        Clear All
                                    </button>
                                )}
                            </div>

                            {terminalsError && (
                                <div className="text-red-400 text-xs bg-red-900/20 p-2 rounded border border-red-800 mb-2">{terminalsError.message}</div>
                            )}

                            {terminals.length === 0 ? (
                                <div className="text-xs text-slate-400 bg-slate-700/30 p-3 rounded border border-dashed border-slate-600 text-center">
                                    <svg className="w-8 h-8 mx-auto mb-1 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                    </svg>
                                    <div>Double-click map to create terminal</div>
                                </div>
                            ) : (
                                <>
                                    {/* Terminal Stats */}
                                    <div className="grid grid-cols-3 gap-1.5 mb-2">
                                        <div className="bg-slate-700/40 rounded p-1.5 text-center">
                                            <div className="text-sm font-bold text-slate-300">
                                                {terminals.filter(t => t.status === 'idle').length}
                                            </div>
                                            <div className="text-[9px] text-slate-500">Idle</div>
                                        </div>
                                        <div className="bg-green-500/10 rounded p-1.5 text-center border border-green-500/20">
                                            <div className="text-sm font-bold text-green-400">
                                                {terminals.filter(t => t.status === 'connected').length}
                                            </div>
                                            <div className="text-[9px] text-green-600">Connected</div>
                                        </div>
                                        <div className="bg-yellow-500/10 rounded p-1.5 text-center border border-yellow-500/20">
                                            <div className="text-sm font-bold text-yellow-400">
                                                {terminals.filter(t => t.status === 'transmitting').length}
                                            </div>
                                            <div className="text-[9px] text-yellow-600">Transmit</div>
                                        </div>
                                    </div>

                                    {/* Terminal List */}
                                    <div className="space-y-1.5 flex-1 overflow-y-auto pr-1 custom-scrollbar">
                                        {terminals.map((terminal) => (
                                            <div
                                                key={terminal.terminalId}
                                                className="bg-slate-700/40 hover:bg-slate-700/60 p-2 rounded border border-slate-600/50 transition-all group"
                                            >
                                                <div className="flex items-center justify-between gap-2">
                                                    <div className="flex items-center gap-2 flex-1 min-w-0">
                                                        <div className={`w-2 h-2 rounded-full ${
                                                            terminal.status === 'connected' ? 'bg-green-400 shadow-lg shadow-green-400/50 animate-pulse' :
                                                            terminal.status === 'transmitting' ? 'bg-yellow-400 shadow-lg shadow-yellow-400/50 animate-pulse' :
                                                            'bg-slate-500'
                                                        }`}></div>
                                                        <div className="flex-1 min-w-0">
                                                            <div className="text-xs font-medium text-white truncate">
                                                                {terminal.terminalId}
                                                            </div>
                                                            <div className="text-[9px] text-slate-400 capitalize mt-0.5">
                                                                {terminal.status || 'idle'}
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <button
                                                        onClick={async () => {
                                                            try {
                                                                await fetch(`http://localhost:5001/api/terminals/${terminal.terminalId}`, {
                                                                    method: 'DELETE'
                                                                });
                                                                const response = await fetch('http://localhost:5001/api/terminals');
                                                                const data = await response.json();
                                                                if (data.terminals) {
                                                                    window.location.reload();
                                                                }
                                                            } catch (error) {
                                                                console.error('Failed to delete terminal:', error);
                                                            }
                                                        }}
                                                        className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 hover:bg-red-500/10 p-1 rounded transition-all"
                                                        title="Delete"
                                                    >
                                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                        </svg>
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Packet Activity */}
                        {activePackets.length > 0 && (
                            <div className="bg-gradient-to-br from-yellow-900/30 to-orange-900/30 rounded-lg p-3 border border-yellow-700/50">
                                <h3 className="text-xs font-semibold text-yellow-300 mb-2 flex items-center gap-1.5">
                                    <svg className="w-3.5 h-3.5 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                    </svg>
                                    Active Packets
                                    <span className="ml-1 px-1.5 py-0.5 bg-yellow-500/30 text-yellow-200 rounded text-[10px] font-bold">{activePackets.length}</span>
                                </h3>
                                <div className="text-xs text-yellow-200/80">
                                    {activePackets.length} packet{activePackets.length > 1 ? 's' : ''} in transit
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default Dashboard;
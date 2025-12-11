// src/components/Sidebar.tsx
import React, { useState, useMemo } from 'react';
import { useNodeStore } from '../state/nodeStore';
import type { NodeDTO } from '../types/NodeTypes';

interface SidebarProps {
    nodes: NodeDTO[];
}

const Sidebar: React.FC<SidebarProps> = ({ nodes }) => {
    const [isOpen, setIsOpen] = useState(true);
    const { selectAndFly, selectedNode, recentlyUpdatedNodes } = useNodeStore();

    const handleNodeClick = (node: NodeDTO) => {
        selectAndFly(node); 
    };

    // Statistics
    const stats = useMemo(() => {
        const healthy = nodes.filter(n => n.healthy).length;
        const avgUtil = nodes.length > 0 
            ? nodes.reduce((sum, n) => sum + (n.resourceUtilization || 0), 0) / nodes.length 
            : 0;
        const byType = nodes.reduce((acc, node) => {
            const type = node.nodeType || 'UNKNOWN';
            acc[type] = (acc[type] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

        return { healthy, total: nodes.length, avgUtil, byType };
    }, [nodes]);

    const getNodeIcon = (type: string) => {
        switch (type) {
            case 'LEO_SATELLITE': return '🛰';
            case 'MEO_SATELLITE': return '🌐';
            case 'GEO_SATELLITE': return '🌍';
            case 'GROUND_STATION': return '📡';
            default: return '•';
        }
    };

    const getUtilColor = (util: number) => {
        if (util < 50) return 'bg-emerald-500';
        if (util < 80) return 'bg-amber-500';
        return 'bg-rose-500';
    };

    return (
        <div className={`bg-slate-900 text-white h-full flex flex-col shadow-xl transition-all duration-200 ${isOpen ? 'w-64' : 'w-12'}`}>
            {/* Header */}
            <div className={`p-2 border-b border-slate-700/50 flex items-center ${!isOpen && 'justify-center'}`}>
                {isOpen && (
                    <div className="flex-1 min-w-0">
                        <h2 className="text-xs font-bold tracking-wide">NETWORK</h2>
                        <p className="text-[10px] text-slate-400">{stats.total} nodes</p>
                    </div>
                )}
                <button 
                    onClick={() => setIsOpen(!isOpen)} 
                    className="p-1 hover:bg-slate-700/50 rounded text-slate-400 hover:text-white transition-colors"
                >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isOpen ? "M15 19l-7-7 7-7" : "M9 5l7 7-7 7"} />
                    </svg>
                </button>
            </div>

            {/* Stats */}
            {isOpen && (
                <div className="p-2 border-b border-slate-700/50 space-y-1.5">
                    <div className="flex gap-1.5">
                        <div className="flex-1 bg-slate-800 rounded p-1.5">
                            <div className="text-[10px] text-slate-400">HEALTH</div>
                            <div className="text-xs font-bold text-emerald-400">{stats.healthy}/{stats.total}</div>
                        </div>
                        <div className="flex-1 bg-slate-800 rounded p-1.5">
                            <div className="text-[10px] text-slate-400">UTIL</div>
                            <div className="text-xs font-bold text-sky-400">{stats.avgUtil.toFixed(0)}%</div>
                        </div>
                    </div>
                    <div className="flex gap-1 text-[10px]">
                        {Object.entries(stats.byType).map(([type, count]) => (
                            <div key={type} className="flex items-center gap-0.5 bg-slate-800 px-1.5 py-0.5 rounded">
                                <span>{getNodeIcon(type)}</span>
                                <span className="text-slate-300 font-semibold">{count}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Nodes List */}
            <div className="flex-1 overflow-y-auto">
                {nodes.map((node) => {
                    const isRecentlyUpdated = recentlyUpdatedNodes.has(node.nodeId);
                    return (
                    <div
                        key={node.nodeId}
                        onClick={() => handleNodeClick(node)}
                        className={`group cursor-pointer transition-colors border-b border-slate-800/50 ${
                            !isOpen && 'flex justify-center'
                        } ${
                            selectedNode?.nodeId === node.nodeId 
                                ? 'bg-sky-600' 
                                : isRecentlyUpdated
                                ? 'bg-amber-900/30 animate-pulse'
                                : 'hover:bg-slate-800'
                        }`}
                    >
                        {isOpen ? (
                            <div className="p-2">
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="text-xs">{getNodeIcon(node.nodeType || '')}</span>
                                    <span className="text-[11px] font-medium truncate flex-1">{node.nodeName}</span>
                                    <div className={`w-1.5 h-1.5 rounded-full ${node.healthy ? 'bg-emerald-400' : 'bg-rose-500'}`} />
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="flex-1 bg-slate-700 rounded-full h-1">
                                        <div 
                                            className={`h-full rounded-full transition-all duration-500 ${
                                                isRecentlyUpdated 
                                                    ? 'shadow-lg shadow-amber-500/50' 
                                                    : ''
                                            } ${getUtilColor(node.resourceUtilization || 0)}`}
                                            style={{ width: `${Math.min(node.resourceUtilization || 0, 100)}%` }}
                                        />
                                    </div>
                                    <span className={`text-[10px] font-semibold w-7 text-right ${
                                        isRecentlyUpdated ? 'text-amber-300' : 'text-slate-300'
                                    }`}>
                                        {(node.resourceUtilization || 0).toFixed(0)}%
                                    </span>
                                </div>
                                {(node.batteryChargePercent !== undefined || node.packetLossRate !== undefined) && (
                                    <div className="flex gap-2 mt-0.5 text-[10px] text-slate-400">
                                        {node.batteryChargePercent !== undefined && (
                                            <span className={node.batteryChargePercent < 20 ? 'text-rose-400' : ''}>
                                                🔋{node.batteryChargePercent.toFixed(0)}%
                                            </span>
                                        )}
                                        {node.packetLossRate !== undefined && node.packetLossRate > 0 && (
                                            <span className={node.packetLossRate > 0.05 ? 'text-amber-400' : ''}>
                                                ↓{(node.packetLossRate * 100).toFixed(1)}%
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="relative p-2">
                                <span className="text-sm">{getNodeIcon(node.nodeType || '')}</span>
                                <div className={`absolute top-1 right-1 w-1.5 h-1.5 rounded-full ${
                                    node.healthy ? 'bg-emerald-400' : 'bg-rose-500'
                                } ${isRecentlyUpdated ? 'animate-ping' : ''}`} />
                                <div className="absolute left-full ml-2 px-2 py-1 bg-slate-800 rounded shadow-lg
                                              invisible opacity-0 group-hover:visible group-hover:opacity-100 transition-all
                                              whitespace-nowrap z-50 text-xs">
                                    <div className="font-semibold">{node.nodeName}</div>
                                    <div className="text-slate-400 text-[10px]">{(node.resourceUtilization || 0).toFixed(0)}% util</div>
                                </div>
                            </div>
                        )}
                    </div>
                    );
                })}
            </div>
        </div>
    );
};

export default Sidebar;
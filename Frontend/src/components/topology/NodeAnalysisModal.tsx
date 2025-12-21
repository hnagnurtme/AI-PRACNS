import React from 'react';
import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ResponsiveContainer,
    Legend,
    Tooltip,
} from 'recharts';
import type { NodeAnalysis } from '../../types/NodeAnalysisTypes';
import type { NodeDTO } from '../../types/NodeTypes';

interface NodeAnalysisModalProps {
    node: NodeDTO | null;
    analysis: NodeAnalysis | null;
    loading: boolean;
    error: Error | null;
    onClose: () => void;
}

const NodeAnalysisModal: React.FC<NodeAnalysisModalProps> = ( {
    node,
    analysis,
    loading,
    error,
    onClose,
} ) => {
    // Prepare radar chart data
    const radarData = React.useMemo( () => {
        if ( !node ) return [];

        const queueRatio =
            node.packetBufferCapacity > 0
                ? ( node.currentPacketCount / node.packetBufferCapacity ) * 100
                : 0;

        return [
            {
                subject: 'Latency',
                value: Math.min( 100, ( 200 - node.nodeProcessingDelayMs ) / 2 ),
                fullMark: 100,
            },
            {
                subject: 'Bandwidth',
                value: Math.min( 100, node.resourceUtilization ),
                fullMark: 100,
            },
            {
                subject: 'Reliability',
                value: Math.max( 0, 100 - node.packetLossRate * 1000 ),
                fullMark: 100,
            },
            {
                subject: 'Queue',
                value: Math.max( 0, 100 - queueRatio ),
                fullMark: 100,
            },
            {
                subject: 'Battery',
                value: node.batteryChargePercent,
                fullMark: 100,
            },
        ];
    }, [ node ] );

    if ( !node ) return null;

    return (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50" onClick={ onClose }>
            <div
                className="bg-cosmic-navy rounded-xl shadow-nebula-lg w-full max-w-7xl max-h-[90vh] overflow-y-auto m-4 border border-white/10 cosmic-scrollbar"
                onClick={ ( e ) => e.stopPropagation() }
            >
                {/* Header */ }
                <div className="sticky top-0 bg-gradient-to-r from-nebula-purple via-nebula-pink to-nebula-cyan p-6 rounded-t-xl">
                    <div className="flex justify-between items-center">
                        <div>
                            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                                <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                                </svg>
                                { node.nodeName }
                            </h2>
                            <p className="text-white/80 text-sm mt-1 font-medium">{ node.nodeType } • Detailed Analysis</p>
                        </div>
                        <button
                            onClick={ onClose }
                            className="text-white hover:text-white/80 text-2xl font-bold w-10 h-10 flex items-center justify-center rounded-full hover:bg-white/20 transition-all"
                        >
                            ×
                        </button>
                    </div>
                </div>

                {/* Node Resource Summary - Horizontal Layout */ }
                <div className="bg-cosmic-dark/50 border-b border-white/10 p-6">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                        <svg className="w-5 h-5 text-nebula-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Node Resources & Performance
                    </h3>
                    <div className="grid grid-cols-5 gap-3">
                        {/* Processing Latency */ }
                        <div className="glass-card p-4">
                            <div className="text-xs font-semibold text-nebula-cyan mb-1 uppercase tracking-wide">Latency</div>
                            <div className="text-2xl font-bold text-white">{ node.nodeProcessingDelayMs.toFixed( 0 ) }</div>
                            <div className="text-xs text-star-silver/70 mt-0.5">ms</div>
                            <div className={ `mt-2 px-2 py-1 rounded text-xs font-semibold text-center ${ node.nodeProcessingDelayMs > 500 ? 'bg-red-500/20 text-red-400' :
                                    node.nodeProcessingDelayMs > 200 ? 'bg-amber-500/20 text-amber-400' :
                                        'bg-emerald-500/20 text-emerald-400'
                                }` }>
                                { node.nodeProcessingDelayMs > 500 ? 'Critical' : node.nodeProcessingDelayMs > 200 ? 'Warning' : 'Normal' }
                            </div>
                        </div>

                        {/* Packet Loss */ }
                        <div className="glass-card p-4">
                            <div className="text-xs font-semibold text-nebula-pink mb-1 uppercase tracking-wide">Packet Loss</div>
                            <div className="text-2xl font-bold text-white">{ ( node.packetLossRate * 100 ).toFixed( 1 ) }</div>
                            <div className="text-xs text-star-silver/70 mt-0.5">%</div>
                            <div className={ `mt-2 px-2 py-1 rounded text-xs font-semibold text-center ${ node.packetLossRate > 0.1 ? 'bg-red-500/20 text-red-400' :
                                    node.packetLossRate > 0.05 ? 'bg-amber-500/20 text-amber-400' :
                                        'bg-emerald-500/20 text-emerald-400'
                                }` }>
                                { node.packetLossRate > 0.1 ? 'Critical' : node.packetLossRate > 0.05 ? 'Warning' : 'Normal' }
                            </div>
                        </div>

                        {/* Queue Buffer */ }
                        <div className="glass-card p-4">
                            <div className="text-xs font-semibold text-nebula-blue mb-1 uppercase tracking-wide">Queue Buffer</div>
                            <div className="text-2xl font-bold text-white">{ node.currentPacketCount }</div>
                            <div className="text-xs text-star-silver/70 mt-0.5">of { node.packetBufferCapacity }</div>
                            <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                                <div
                                    className={ `h-2 rounded-full ${ ( node.currentPacketCount / node.packetBufferCapacity ) > 0.9 ? 'bg-red-500' :
                                            ( node.currentPacketCount / node.packetBufferCapacity ) > 0.7 ? 'bg-amber-500' : 'bg-emerald-500'
                                        }` }
                                    style={ { width: `${ Math.min( 100, ( node.currentPacketCount / node.packetBufferCapacity ) * 100 ) }%` } }
                                ></div>
                            </div>
                        </div>

                        {/* CPU/Memory */ }
                        <div className="glass-card p-4">
                            <div className="text-xs font-semibold text-nebula-purple mb-1 uppercase tracking-wide">CPU/Memory</div>
                            <div className="text-2xl font-bold text-white">{ node.resourceUtilization.toFixed( 0 ) }</div>
                            <div className="text-xs text-star-silver/70 mt-0.5">%</div>
                            <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                                <div
                                    className={ `h-2 rounded-full ${ node.resourceUtilization > 90 ? 'bg-red-500' :
                                            node.resourceUtilization > 70 ? 'bg-amber-500' : 'bg-emerald-500'
                                        }` }
                                    style={ { width: `${ node.resourceUtilization }%` } }
                                ></div>
                            </div>
                        </div>

                        {/* Battery */ }
                        <div className="glass-card p-4">
                            <div className="text-xs font-semibold text-star-gold mb-1 uppercase tracking-wide">Battery</div>
                            <div className="text-2xl font-bold text-white">{ node.batteryChargePercent.toFixed( 0 ) }</div>
                            <div className="text-xs text-star-silver/70 mt-0.5">%</div>
                            <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                                <div
                                    className={ `h-2 rounded-full ${ node.batteryChargePercent > 50 ? 'bg-emerald-500' :
                                            node.batteryChargePercent > 20 ? 'bg-amber-500' : 'bg-red-500'
                                        }` }
                                    style={ { width: `${ node.batteryChargePercent }%` } }
                                ></div>
                            </div>
                        </div>
                    </div>

                    {/* Additional Resource Details - Horizontal */ }
                    <div className="grid grid-cols-4 gap-3 mt-4">
                        <div className="glass-card p-3">
                            <div className="text-xs text-nebula-cyan font-semibold mb-1">Bandwidth</div>
                            <div className="text-lg font-bold text-white">{ ( node as any ).bandwidthMbps?.toFixed( 0 ) || 'N/A' } <span className="text-sm text-star-silver/70">Mbps</span></div>
                        </div>
                        <div className="glass-card p-3">
                            <div className="text-xs text-nebula-cyan font-semibold mb-1">Signal Strength</div>
                            <div className="text-lg font-bold text-white">{ ( node as any ).signalStrengthDbm?.toFixed( 0 ) || 'N/A' } <span className="text-sm text-star-silver/70">dBm</span></div>
                        </div>
                        <div className="glass-card p-3">
                            <div className="text-xs text-nebula-cyan font-semibold mb-1">Status</div>
                            <div className={ `text-sm font-bold px-2 py-1 rounded inline-block ${ node.isOperational ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                                }` }>
                                { node.isOperational ? 'Online' : 'Offline' }
                            </div>
                        </div>
                        <div className="glass-card p-3">
                            <div className="text-xs text-nebula-cyan font-semibold mb-1">Position</div>
                            <div className="text-xs text-star-silver">
                                Lat: { node.position.latitude.toFixed( 1 ) }°<br />
                                Lon: { node.position.longitude.toFixed( 1 ) }°<br />
                                Alt: { ( node.position.altitude / 1000 ).toFixed( 0 ) } km
                            </div>
                        </div>
                    </div>
                </div>

                {/* Content */ }
                <div className="p-6 space-y-6">
                    { loading && (
                        <div className="flex items-center justify-center py-12">
                            <div className="animate-spin rounded-full h-12 w-12 border-4 border-nebula-purple/30 border-t-nebula-purple"></div>
                            <span className="ml-4 text-star-silver">Loading analysis...</span>
                        </div>
                    ) }

                    { error && (
                        <div className="glass-card border-red-500/30 p-4">
                            <span className="text-red-400">Error: { error.message }</span>
                        </div>
                    ) }

                    { !loading && !error && analysis && (
                        <>
                            {/* Radar Chart - Pentagon */ }
                            <div className="glass-card p-6">
                                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <svg className="w-5 h-5 text-nebula-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                    </svg>
                                    Performance Score Breakdown
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart data={ radarData }>
                                            <PolarGrid stroke="#7c3aed" strokeOpacity={ 0.3 } />
                                            <PolarAngleAxis dataKey="subject" tick={ { fontSize: 12, fill: '#94a3b8' } } />
                                            <PolarRadiusAxis angle={ 90 } domain={ [ 0, 100 ] } tick={ { fontSize: 10, fill: '#94a3b8' } } />
                                            <Radar
                                                name="Performance"
                                                dataKey="value"
                                                stroke="#7c3aed"
                                                fill="#7c3aed"
                                                fillOpacity={ 0.5 }
                                            />
                                            <Tooltip contentStyle={ { backgroundColor: '#1e1b4b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' } } />
                                            <Legend />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Best Links */ }
                            { analysis.bestLinks.length > 0 && (
                                <div className="glass-card p-5 border-emerald-500/30">
                                    <h3 className="text-lg font-bold text-emerald-400 mb-4 flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                                        </svg>
                                        Best Neighbor Links
                                        <span className="text-sm font-normal text-emerald-400/70 bg-emerald-500/10 px-2 py-0.5 rounded-full">
                                            { analysis.bestLinks.length } optimal connections
                                        </span>
                                    </h3>
                                    <div className="space-y-2 max-h-64 overflow-y-auto cosmic-scrollbar">
                                        { analysis.bestLinks.map( ( link, index ) => (
                                            <div
                                                key={ link.nodeId }
                                                className="bg-white/5 rounded-lg p-4 border border-emerald-500/20 hover:bg-white/10 transition-all"
                                            >
                                                <div className="flex justify-between items-start">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <span className="text-xs font-bold text-white bg-gradient-to-r from-emerald-500 to-teal-500 px-2.5 py-1 rounded-full">
                                                                #{ index + 1 }
                                                            </span>
                                                            <span className="font-bold text-white">{ link.nodeName }</span>
                                                        </div>
                                                        <div className="flex items-center gap-3">
                                                            <div className={ `px-3 py-1 rounded-full text-xs font-bold ${ link.quality === 'excellent' ? 'bg-emerald-500/20 text-emerald-400' :
                                                                    link.quality === 'good' ? 'bg-blue-500/20 text-blue-400' :
                                                                        link.quality === 'fair' ? 'bg-amber-500/20 text-amber-400' : 'bg-red-500/20 text-red-400'
                                                                }` }>
                                                                { link.quality.toUpperCase() }
                                                            </div>
                                                            <div className="text-sm">
                                                                <span className="text-star-silver/70">Score:</span>
                                                                <span className="font-bold text-emerald-400 ml-1">{ link.score.toFixed( 1 ) }/100</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right space-y-1.5 ml-4 bg-white/5 rounded-lg p-3 border border-white/10">
                                                        <div className="font-bold text-star-silver flex items-center justify-end gap-1">
                                                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                                            </svg>
                                                            { link.distance.toFixed( 0 ) } km
                                                        </div>
                                                        <div className="text-xs text-star-silver/70">Latency: <span className="font-semibold text-star-silver">{ link.latency.toFixed( 0 ) }ms</span></div>
                                                        <div className="text-xs text-star-silver/70">Bandwidth: <span className="font-semibold text-star-silver">{ link.bandwidth.toFixed( 0 ) }Mbps</span></div>
                                                        <div className="text-xs text-star-silver/70">Loss: <span className="font-semibold text-star-silver">{ ( link.packetLoss * 100 ).toFixed( 2 ) }%</span></div>
                                                    </div>
                                                </div>
                                            </div>
                                        ) ) }
                                    </div>
                                </div>
                            ) }

                            {/* Upcoming Satellites */ }
                            { analysis.upcomingSatellites.length > 0 && (
                                <div className="glass-card p-5 border-nebula-purple/30">
                                    <h3 className="text-lg font-bold text-nebula-purple mb-4 flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                                        </svg>
                                        Incoming Satellites
                                        <span className="text-sm font-normal text-nebula-purple/70 bg-nebula-purple/10 px-2 py-0.5 rounded-full">
                                            { analysis.upcomingSatellites.length } approaching
                                        </span>
                                    </h3>
                                    <div className="space-y-2 max-h-64 overflow-y-auto cosmic-scrollbar">
                                        { analysis.upcomingSatellites.map( ( sat, index ) => {
                                            const minutes = Math.floor( sat.estimatedArrivalIn / 60 );
                                            const seconds = sat.estimatedArrivalIn % 60;
                                            const hours = Math.floor( minutes / 60 );
                                            const remainingMinutes = minutes % 60;

                                            let timeDisplay = '';
                                            if ( hours > 0 ) {
                                                timeDisplay = `${ hours }h ${ remainingMinutes }m`;
                                            } else if ( minutes > 0 ) {
                                                timeDisplay = `${ minutes }m ${ seconds }s`;
                                            } else {
                                                timeDisplay = `${ seconds }s`;
                                            }

                                            return (
                                                <div
                                                    key={ sat.nodeId }
                                                    className="bg-white/5 rounded p-3 border border-nebula-purple/20 hover:bg-white/10 transition-all"
                                                >
                                                    <div className="flex justify-between items-start">
                                                        <div className="flex-1">
                                                            <div className="flex items-center gap-2 mb-2">
                                                                <span className="text-xs font-bold text-white bg-gradient-to-r from-nebula-purple to-nebula-pink px-2.5 py-1 rounded-full">
                                                                    #{ index + 1 }
                                                                </span>
                                                                <span className="font-bold text-white">{ sat.nodeName }</span>
                                                            </div>
                                                            <div className="text-sm text-star-silver/70 space-y-1">
                                                                <div className="font-medium text-star-silver">{ sat.nodeType }</div>
                                                                <div className="flex items-center gap-1">
                                                                    { sat.willBeInRange ? (
                                                                        <span className="text-emerald-400 text-xs font-semibold bg-emerald-500/10 px-2 py-0.5 rounded-md">✅ Will be in range</span>
                                                                    ) : (
                                                                        <span className="text-amber-400 text-xs font-semibold bg-amber-500/10 px-2 py-0.5 rounded-md">⚠️ May be out of range</span>
                                                                    ) }
                                                                </div>
                                                            </div>
                                                        </div>
                                                        <div className="text-right space-y-2 ml-4 bg-nebula-purple/10 rounded-lg p-3 border border-nebula-purple/20">
                                                            <div className="font-bold text-nebula-purple text-lg flex items-center justify-end gap-1">
                                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                                </svg>
                                                                ETA: { timeDisplay }
                                                            </div>
                                                            <div className="text-xs space-y-1 pt-2 border-t border-nebula-purple/20">
                                                                <div className="text-star-silver/70">Est. Latency: <span className="font-semibold text-star-silver">{ sat.estimatedLatency }ms</span></div>
                                                                <div className="text-star-silver/70">Est. Bandwidth: <span className="font-semibold text-star-silver">{ sat.estimatedBandwidth }Mbps</span></div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        } ) }
                                    </div>
                                </div>
                            ) }

                            {/* Degrading Nodes */ }
                            { analysis.degradingNodes.length > 0 && (
                                <div className="glass-card p-5 border-red-500/30">
                                    <h3 className="text-lg font-bold text-red-400 mb-4 flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                        </svg>
                                        Performance Degradation Alerts
                                        <span className="text-sm font-normal text-red-400/70 bg-red-500/10 px-2 py-0.5 rounded-full">
                                            { analysis.degradingNodes.length } nodes at risk
                                        </span>
                                    </h3>
                                    <div className="space-y-2 max-h-48 overflow-y-auto cosmic-scrollbar">
                                        { analysis.degradingNodes.map( ( degrading ) => (
                                            <div
                                                key={ degrading.nodeId }
                                                className={ `bg-white/5 rounded-lg p-4 border hover:bg-white/10 transition-all ${ degrading.severity === 'critical'
                                                        ? 'border-red-500/30'
                                                        : degrading.severity === 'warning'
                                                            ? 'border-amber-500/30'
                                                            : 'border-orange-500/30'
                                                    }` }
                                            >
                                                <div className="flex justify-between items-start">
                                                    <div className="flex-1">
                                                        <div className="font-bold text-white mb-2">{ degrading.nodeName }</div>
                                                        <div className={ `inline-block px-3 py-1 rounded-full text-xs font-bold mb-2 ${ degrading.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                                                                degrading.severity === 'warning' ? 'bg-amber-500/20 text-amber-400' :
                                                                    'bg-orange-500/20 text-orange-400'
                                                            }` }>
                                                            { degrading.severity.toUpperCase() } SEVERITY
                                                        </div>
                                                        <div className="text-xs text-star-silver/70 font-medium">
                                                            Causes: <span className="text-star-silver">{ degrading.degradationReason.join( ' • ' ) }</span>
                                                        </div>
                                                    </div>
                                                    <div className="text-right space-y-1.5 ml-4 bg-white/5 rounded-lg p-3 border border-white/10">
                                                        <div className="font-bold text-red-400 text-sm">
                                                            ⏰ In { Math.floor( degrading.predictedDegradationIn / 60 ) } min
                                                        </div>
                                                        <div className="text-xs text-star-silver/70 pt-2 border-t border-white/10 space-y-1">
                                                            <div>Latency: <span className="font-semibold text-star-silver">{ degrading.currentMetrics.latency.toFixed( 0 ) }ms</span></div>
                                                            <div>Load: <span className="font-semibold text-star-silver">{ degrading.currentMetrics.utilization.toFixed( 0 ) }%</span></div>
                                                            <div>Battery: <span className="font-semibold text-star-silver">{ degrading.currentMetrics.battery.toFixed( 0 ) }%</span></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ) ) }
                                    </div>
                                </div>
                            ) }

                            {/* No predictions message */ }
                            { analysis.bestLinks.length === 0 &&
                                analysis.upcomingSatellites.length === 0 &&
                                analysis.degradingNodes.length === 0 && (
                                    <div className="glass-card p-8 text-center">
                                        <svg className="w-16 h-16 text-star-silver/50 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <p className="text-star-silver font-medium">No analysis data available for this node</p>
                                        <p className="text-star-silver/70 text-sm mt-1">All metrics are within normal ranges</p>
                                    </div>
                                ) }
                        </>
                    ) }
                </div>
            </div>
        </div>
    );
};

export default NodeAnalysisModal;

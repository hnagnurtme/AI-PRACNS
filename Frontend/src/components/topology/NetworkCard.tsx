import React from 'react';
import type { NetworkStatistics } from '../../types/NetworkTopologyTypes';

interface NetworkCardProps {
    statistics: NetworkStatistics;
    networkName?: string;
    onClick?: () => void;
}

const NetworkCard: React.FC<NetworkCardProps> = ( { statistics, networkName = 'SAGIN Network', onClick } ) => {
    const getHealthColor = ( health: string ) => {
        switch ( health ) {
            case 'healthy':
                return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
            case 'degraded':
                return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
            case 'critical':
                return 'bg-red-500/20 text-red-400 border-red-500/30';
            default:
                return 'bg-white/10 text-star-silver border-white/20';
        }
    };

    const getHealthIcon = ( health: string ) => {
        switch ( health ) {
            case 'healthy':
                return '✅';
            case 'degraded':
                return '⚠️';
            case 'critical':
                return '🔴';
            default:
                return '❓';
        }
    };

    return (
        <div
            onClick={ onClick }
            className={ `glass-card p-6 cursor-pointer transition-all hover:shadow-nebula ${ onClick ? 'hover:scale-[1.02]' : ''
                }` }
        >
            <div className="flex justify-between items-start mb-5">
                <h3 className="text-xl font-bold bg-gradient-to-r from-nebula-purple via-nebula-pink to-nebula-cyan bg-clip-text text-transparent">{ networkName }</h3>
                <div className={ `px-3 py-1.5 rounded-full text-xs font-bold border ${ getHealthColor( statistics.networkHealth ) }` }>
                    { getHealthIcon( statistics.networkHealth ) } { statistics.networkHealth.toUpperCase() }
                </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="bg-nebula-blue/10 rounded-xl p-4 border border-nebula-blue/20">
                    <div className="text-xs font-semibold text-nebula-blue mb-1 uppercase tracking-wide">Network Nodes</div>
                    <div className="text-3xl font-bold text-nebula-blue">{ statistics.totalNodes }</div>
                    <div className="text-xs text-nebula-blue/70 mt-1 font-medium">
                        { statistics.activeNodes } online • { statistics.totalNodes - statistics.activeNodes } offline
                    </div>
                </div>

                <div className="bg-nebula-purple/10 rounded-xl p-4 border border-nebula-purple/20">
                    <div className="text-xs font-semibold text-nebula-purple mb-1 uppercase tracking-wide">User Terminals</div>
                    <div className="text-3xl font-bold text-nebula-purple">{ statistics.totalTerminals }</div>
                    <div className="text-xs text-nebula-purple/70 mt-1 font-medium">
                        { statistics.connectedTerminals } connected • { statistics.totalTerminals - statistics.connectedTerminals } idle
                    </div>
                </div>

                <div className="bg-emerald-500/10 rounded-xl p-4 border border-emerald-500/20">
                    <div className="text-xs font-semibold text-emerald-400 mb-1 uppercase tracking-wide">Active Links</div>
                    <div className="text-3xl font-bold text-emerald-400">{ statistics.activeConnections }</div>
                    <div className="text-xs text-emerald-400/70 mt-1 font-medium">
                        { statistics.totalConnections } total • { ( ( statistics.activeConnections / statistics.totalConnections ) * 100 ).toFixed( 0 ) }% active
                    </div>
                </div>

                <div className="bg-star-gold/10 rounded-xl p-4 border border-star-gold/20">
                    <div className="text-xs font-semibold text-star-gold mb-1 uppercase tracking-wide">Total Bandwidth</div>
                    <div className="text-3xl font-bold text-star-gold">{ statistics.totalBandwidth.toFixed( 0 ) }</div>
                    <div className="text-xs text-star-gold/70 mt-1 font-medium">
                        Mbps • Avg { ( statistics.totalBandwidth / statistics.activeNodes ).toFixed( 1 ) } per node
                    </div>
                </div>
            </div>

            <div className="nebula-divider mb-4"></div>

            <div className="space-y-3">
                <div className="flex justify-between items-center text-sm">
                    <span className="text-star-silver font-medium">Avg Network Latency</span>
                    <span className={ `font-bold px-2 py-1 rounded ${ statistics.averageLatency > 200 ? 'text-red-400 bg-red-500/10' :
                            statistics.averageLatency > 100 ? 'text-amber-400 bg-amber-500/10' :
                                'text-emerald-400 bg-emerald-500/10'
                        }` }>{ statistics.averageLatency.toFixed( 1 ) } ms</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                    <span className="text-star-silver font-medium">Avg Packet Loss Rate</span>
                    <span className={ `font-bold px-2 py-1 rounded ${ statistics.averagePacketLoss > 0.05 ? 'text-red-400 bg-red-500/10' :
                            statistics.averagePacketLoss > 0.02 ? 'text-amber-400 bg-amber-500/10' :
                                'text-emerald-400 bg-emerald-500/10'
                        }` }>{ ( statistics.averagePacketLoss * 100 ).toFixed( 3 ) }%</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                    <span className="text-star-silver font-medium">Resource Utilization</span>
                    <span className={ `font-bold px-2 py-1 rounded ${ statistics.utilizationRate > 80 ? 'text-red-400 bg-red-500/10' :
                            statistics.utilizationRate > 60 ? 'text-amber-400 bg-amber-500/10' :
                                'text-emerald-400 bg-emerald-500/10'
                        }` }>{ statistics.utilizationRate.toFixed( 1 ) }%</span>
                </div>
            </div>
        </div>
    );
};

export default NetworkCard;

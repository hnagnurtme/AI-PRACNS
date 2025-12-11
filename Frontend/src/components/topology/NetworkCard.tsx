import React from 'react';
import type { NetworkStatistics } from '../../types/NetworkTopologyTypes';

interface NetworkCardProps {
    statistics: NetworkStatistics;
    networkName?: string;
    onClick?: () => void;
}

const NetworkCard: React.FC<NetworkCardProps> = ({ statistics, networkName = 'SAGIN Network', onClick }) => {
    const getHealthColor = (health: string) => {
        switch (health) {
            case 'healthy':
                return 'bg-green-100 text-green-800 border-green-300';
            case 'degraded':
                return 'bg-yellow-100 text-yellow-800 border-yellow-300';
            case 'critical':
                return 'bg-red-100 text-red-800 border-red-300';
            default:
                return 'bg-gray-100 text-gray-800 border-gray-300';
        }
    };

    const getHealthIcon = (health: string) => {
        switch (health) {
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
            onClick={onClick}
            className={`bg-gradient-to-br from-white to-violet-50/30 rounded-xl shadow-xl border border-violet-200 p-6 cursor-pointer transition-all hover:shadow-2xl ${
                onClick ? 'hover:scale-[1.02]' : ''
            }`}
        >
            <div className="flex justify-between items-start mb-5">
                <h3 className="text-xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent">{networkName}</h3>
                <div className={`px-3 py-1.5 rounded-full text-xs font-bold border shadow-sm ${getHealthColor(statistics.networkHealth)}`}>
                    {getHealthIcon(statistics.networkHealth)} {statistics.networkHealth.toUpperCase()}
                </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100/50 rounded-xl p-4 border border-blue-200/50 shadow-sm">
                    <div className="text-xs font-semibold text-blue-700 mb-1 uppercase tracking-wide">Network Nodes</div>
                    <div className="text-3xl font-bold text-blue-600">{statistics.totalNodes}</div>
                    <div className="text-xs text-blue-600 mt-1 font-medium">
                        {statistics.activeNodes} online • {statistics.totalNodes - statistics.activeNodes} offline
                    </div>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100/50 rounded-xl p-4 border border-purple-200/50 shadow-sm">
                    <div className="text-xs font-semibold text-purple-700 mb-1 uppercase tracking-wide">User Terminals</div>
                    <div className="text-3xl font-bold text-purple-600">{statistics.totalTerminals}</div>
                    <div className="text-xs text-purple-600 mt-1 font-medium">
                        {statistics.connectedTerminals} connected • {statistics.totalTerminals - statistics.connectedTerminals} idle
                    </div>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-green-100/50 rounded-xl p-4 border border-green-200/50 shadow-sm">
                    <div className="text-xs font-semibold text-green-700 mb-1 uppercase tracking-wide">Active Links</div>
                    <div className="text-3xl font-bold text-green-600">{statistics.activeConnections}</div>
                    <div className="text-xs text-green-600 mt-1 font-medium">
                        {statistics.totalConnections} total • {((statistics.activeConnections/statistics.totalConnections)*100).toFixed(0)}% active
                    </div>
                </div>

                <div className="bg-gradient-to-br from-orange-50 to-orange-100/50 rounded-xl p-4 border border-orange-200/50 shadow-sm">
                    <div className="text-xs font-semibold text-orange-700 mb-1 uppercase tracking-wide">Total Bandwidth</div>
                    <div className="text-3xl font-bold text-orange-600">{statistics.totalBandwidth.toFixed(0)}</div>
                    <div className="text-xs text-orange-600 mt-1 font-medium">
                        Mbps • Avg {(statistics.totalBandwidth/statistics.activeNodes).toFixed(1)} per node
                    </div>
                </div>
            </div>

            <div className="border-t border-violet-200 pt-4 space-y-3">
                <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-600 font-medium">Avg Network Latency</span>
                    <span className={`font-bold px-2 py-1 rounded ${
                        statistics.averageLatency > 200 ? 'text-red-600 bg-red-50' :
                        statistics.averageLatency > 100 ? 'text-yellow-600 bg-yellow-50' :
                        'text-green-600 bg-green-50'
                    }`}>{statistics.averageLatency.toFixed(1)} ms</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-600 font-medium">Avg Packet Loss Rate</span>
                    <span className={`font-bold px-2 py-1 rounded ${
                        statistics.averagePacketLoss > 0.05 ? 'text-red-600 bg-red-50' :
                        statistics.averagePacketLoss > 0.02 ? 'text-yellow-600 bg-yellow-50' :
                        'text-green-600 bg-green-50'
                    }`}>{(statistics.averagePacketLoss * 100).toFixed(3)}%</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-600 font-medium">Resource Utilization</span>
                    <span className={`font-bold px-2 py-1 rounded ${
                        statistics.utilizationRate > 80 ? 'text-red-600 bg-red-50' :
                        statistics.utilizationRate > 60 ? 'text-yellow-600 bg-yellow-50' :
                        'text-green-600 bg-green-50'
                    }`}>{statistics.utilizationRate.toFixed(1)}%</span>
                </div>
            </div>
        </div>
    );
};

export default NetworkCard;


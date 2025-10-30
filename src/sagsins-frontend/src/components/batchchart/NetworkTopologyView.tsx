import { useMemo } from 'react';
import { Activity} from 'lucide-react';
import type { NetworkBatch, NodeCongestion } from '../../types/ComparisonTypes';
import { NodeCongestionCard } from './NodeCongestionCard';
const safeFixed = (value: number | undefined | null, decimals: number = 1): string => {
    if (value === undefined || value === null || isNaN(value) || !isFinite(value)) {
        return '0';
    }
    return value.toFixed(decimals);
};

export const NetworkTopologyView = ({
    congestionMap,
    selectedNode,
    onSelectNode
}: {
    congestionMap: NodeCongestion[];
    selectedNode: string | null;
    onSelectNode: (nodeId: string) => void;
}) => {
    return (
        <div className="bg-white rounded-lg border border-gray-300 p-6 w-full">
            <div className="flex items-center gap-2 mb-4">
                <Activity className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-bold text-gray-800">Network Nodes Congestion</h3>
            </div>

            {congestionMap.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                    No nodes available
                </div>
            ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {congestionMap.map(node => (
                        <NodeCongestionCard
                            key={node.nodeId}
                            node={node}
                            onSelectNode={() => onSelectNode(node.nodeId)}
                            isSelected={selectedNode === node.nodeId}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

// Component: Batch Statistics
export const BatchStatistics = ({ batch, congestionMap }: { batch: NetworkBatch; congestionMap: NodeCongestion[] }) => {
    const stats = useMemo(() => {
        const totalNodes = congestionMap.length;
        const highCongestion = congestionMap.filter(n => n.avgBandwidthUtil > 0.8).length;
        const mediumCongestion = congestionMap.filter(n => n.avgBandwidthUtil > 0.5 && n.avgBandwidthUtil <= 0.8).length;

        let dijkstraTotal = 0, rlTotal = 0;
        let validPackets = 0;

        batch.packets.forEach(pair => {
            const dijkstraLatency = pair.dijkstraPacket?.analysisData?.totalLatencyMs;
            const rlLatency = pair.rlpacket?.analysisData?.totalLatencyMs;

            if (dijkstraLatency !== undefined && !isNaN(dijkstraLatency)) {
                dijkstraTotal += dijkstraLatency;
                validPackets++;
            }
            if (rlLatency !== undefined && !isNaN(rlLatency)) {
                rlTotal += rlLatency;
            }
        });

        const avgDijkstra = validPackets > 0 ? dijkstraTotal / validPackets : 0;
        const avgRL = validPackets > 0 ? rlTotal / validPackets : 0;
        const improvement = avgDijkstra > 0 ? ((dijkstraTotal - rlTotal) / dijkstraTotal * 100) : 0;

        return {
            totalNodes,
            highCongestion,
            mediumCongestion,
            avgDijkstra,
            avgRL,
            improvement
        };
    }, [batch, congestionMap]);

    return (
        <div className="bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 rounded-lg p-6 border border-gray-300 w-full">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Batch: {batch.batchId}</h2>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Total Nodes</div>
                    <div className="text-2xl font-bold text-gray-800">{stats.totalNodes}</div>
                </div>

                <div className="bg-white rounded-lg p-4 border border-red-200">
                    <div className="text-xs text-gray-500 mb-1">High Congestion</div>
                    <div className="text-2xl font-bold text-red-600">{stats.highCongestion}</div>
                </div>

                <div className="bg-white rounded-lg p-4 border border-yellow-200">
                    <div className="text-xs text-gray-500 mb-1">Medium Congestion</div>
                    <div className="text-2xl font-bold text-yellow-600">{stats.mediumCongestion}</div>
                </div>

                <div className="bg-white rounded-lg p-4 border border-blue-200">
                    <div className="text-xs text-gray-500 mb-1">Dijkstra Avg</div>
                    <div className="text-2xl font-bold text-blue-600">
                        {safeFixed(stats.avgDijkstra, 1)}<span className="text-sm">ms</span>
                    </div>
                </div>

                <div className="bg-white rounded-lg p-4 border border-purple-200">
                    <div className="text-xs text-gray-500 mb-1">RL Avg</div>
                    <div className="text-2xl font-bold text-purple-600">
                        {safeFixed(stats.avgRL, 1)}<span className="text-sm">ms</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
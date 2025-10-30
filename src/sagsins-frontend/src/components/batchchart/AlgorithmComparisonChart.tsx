import { useMemo } from 'react';
import type { NetworkBatch } from '../../types/ComparisonTypes';
import { TrendingUp } from 'lucide-react';

// Utility function to safely format numbers
const safeFixed = (value: number | undefined | null, decimals: number = 1): string => {
    if (value === undefined || value === null || isNaN(value) || !isFinite(value)) {
        return '0';
    }
    return value.toFixed(decimals);
};
export const AlgorithmComparisonChart = ({ batch }: { batch: NetworkBatch }) => {
    const comparison = useMemo(() => {
        let dijkstraWins = 0, rlWins = 0, ties = 0;
        let dijkstraDropped = 0, rlDropped = 0;

        batch.packets.forEach(pair => {
            const dijkstraLatency = pair.dijkstraPacket?.analysisData?.totalLatencyMs;
            const rlLatency = pair.rlpacket?.analysisData?.totalLatencyMs;

            if (dijkstraLatency !== undefined && rlLatency !== undefined && 
                !isNaN(dijkstraLatency) && !isNaN(rlLatency)) {
                const diff = dijkstraLatency - rlLatency;
                if (Math.abs(diff) < 1) ties++;
                else if (diff > 0) rlWins++;
                else dijkstraWins++;
            }

            if (pair.dijkstraPacket?.dropped) dijkstraDropped++;
            if (pair.rlpacket?.dropped) rlDropped++;
        });

        return { dijkstraWins, rlWins, ties, dijkstraDropped, rlDropped };
    }, [batch]);

    const totalPairs = batch.totalPairPackets || 1; // Avoid division by zero

    return (
        <div className="bg-white rounded-lg border border-gray-300 p-6 w-full">
            <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-bold text-gray-800">Algorithm Performance</h3>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                    <div className="text-sm text-gray-600 mb-3">Win Rate</div>
                    <div className="space-y-2">
                        <div className="flex items-center gap-3">
                            <div className="w-24 text-sm text-blue-600 font-medium">Dijkstra</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                                <div
                                    className="bg-blue-500 h-full flex items-center justify-end pr-2"
                                    style={{ width: `${(comparison.dijkstraWins / totalPairs) * 100}%` }}
                                >
                                    {comparison.dijkstraWins > 0 && (
                                        <span className="text-xs text-white font-bold">{comparison.dijkstraWins}</span>
                                    )}
                                </div>
                            </div>
                            <span className="text-xs text-gray-500 w-12 text-right">
                                {safeFixed((comparison.dijkstraWins / totalPairs) * 100, 0)}%
                            </span>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="w-24 text-sm text-purple-600 font-medium">RL</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                                <div
                                    className="bg-purple-500 h-full flex items-center justify-end pr-2"
                                    style={{ width: `${(comparison.rlWins / totalPairs) * 100}%` }}
                                >
                                    {comparison.rlWins > 0 && (
                                        <span className="text-xs text-white font-bold">{comparison.rlWins}</span>
                                    )}
                                </div>
                            </div>
                            <span className="text-xs text-gray-500 w-12 text-right">
                                {safeFixed((comparison.rlWins / totalPairs) * 100, 0)}%
                            </span>
                        </div>
                    </div>
                </div>

                <div>
                    <div className="text-sm text-gray-600 mb-3">Dropped Packets</div>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-blue-50 rounded">
                            <span className="text-sm text-blue-700">Dijkstra</span>
                            <span className="text-xl font-bold text-blue-700">{comparison.dijkstraDropped}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-purple-50 rounded">
                            <span className="text-sm text-purple-700">RL</span>
                            <span className="text-xl font-bold text-purple-700">{comparison.rlDropped}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
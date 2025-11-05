import { useMemo } from 'react';
import { Layers } from 'lucide-react';
import type { NetworkBatch, NodeCongestion, Packet } from '../../types/ComparisonTypes';

const safeFixed = (value: number | undefined | null, decimals: number = 1): string => {
    if (value === undefined || value === null || isNaN(value) || !isFinite(value)) {
        return '0';
    }
    return value.toFixed(decimals);
};

// Component: Packet Flow Through Node
export const PacketFlowDetail = ({ node, batch }: { node: NodeCongestion; batch: NetworkBatch }) => {
    const packetsAtNode = useMemo(() => {
        const packets: Array<{ packet: Packet; algorithm: string; pairIndex: number }> = [];

        batch.packets.forEach((pair, pairIdx) => {
            // Use rlPacket to match server naming and guard against null packets
            [pair.dijkstraPacket, pair.rlPacket].forEach(packet => {
                if (!packet) return; // packet can be null when one side is absent

                const hasNode = (packet.hopRecords || []).some(
                    hop => hop.fromNodeId === node.nodeId || hop.toNodeId === node.nodeId
                );
                if (hasNode) {
                    packets.push({
                        packet,
                        algorithm: packet.useRL ? 'RL' : 'Dijkstra',
                        pairIndex: pairIdx
                    });
                }
            });
        });

        return packets;
    }, [node, batch]);

    return (
        <div className="bg-white rounded-lg border-2 border-blue-400 p-6 w-full">
            <div className="flex items-center gap-3 mb-4">
                <Layers className="w-6 h-6 text-blue-600" />
                <div>
                    <h3 className="text-xl font-bold text-gray-800">{node.nodeId}</h3>
                    <p className="text-sm text-gray-600">{packetsAtNode.length} packets routed through this node</p>
                </div>
            </div>

            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                {packetsAtNode.length === 0 ? (
                    <div className="text-center text-gray-500 py-8">
                        No packets found for this node
                    </div>
                ) : (
                    packetsAtNode.map((item, idx) => {
                        const hopAtNode = item.packet.hopRecords.find(
                            h => h.fromNodeId === node.nodeId
                        );

                        return (
                            <div
                                key={idx}
                                className={`p-3 rounded border ${item.algorithm === 'Dijkstra'
                                    ? 'bg-blue-50 border-blue-200'
                                    : 'bg-purple-50 border-purple-200'
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-mono text-gray-600">
                                        Pair #{item.pairIndex + 1} - {item.packet.packetId.substring(0, 12)}...
                                    </span>
                                    <span className={`text-xs font-semibold ${item.algorithm === 'Dijkstra' ? 'text-blue-600' : 'text-purple-600'
                                        }`}>
                                        {item.algorithm}
                                    </span>
                                </div>

                                {hopAtNode && (
                                    <div className="grid grid-cols-3 gap-2 text-xs">
                                        <div>
                                            <div className="text-gray-500">Next Hop</div>
                                            <div className="font-semibold">{hopAtNode.toNodeId}</div>
                                        </div>
                                        <div>
                                            <div className="text-gray-500">Queue Size</div>
                                            <div className="font-semibold">{hopAtNode.fromNodeBufferState.queueSize || 0}</div>
                                        </div>
                                        <div>
                                            <div className="text-gray-500">Bandwidth</div>
                                            <div className="font-semibold">
                                                {safeFixed(hopAtNode.fromNodeBufferState.bandwidthUtilization * 100, 0)}%
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <div className="mt-2 text-xs text-gray-600">
                                    Route: {item.packet.stationSource} → {item.packet.stationDest}
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};
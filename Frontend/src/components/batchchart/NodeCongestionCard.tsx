import { useMemo } from 'react';
import { AlertTriangle, Radio, Layers } from 'lucide-react';
import type { NetworkBatch, NodeCongestion, Packet } from '../../types/ComparisonTypes';

// Utility function to safely format numbers
const safeFixed = ( value: number | undefined | null, decimals: number = 1 ): string => {
    if ( value === undefined || value === null || isNaN( value ) || !isFinite( value ) ) {
        return '0';
    }
    return value.toFixed( decimals );
};

// Component: Node Congestion Card
export const NodeCongestionCard = ( { node, onSelectNode, isSelected }: {
    node: NodeCongestion;
    onSelectNode: () => void;
    isSelected: boolean;
} ) => {
    const congestionLevel = node.avgBandwidthUtil > 0.8 ? 'high' : node.avgBandwidthUtil > 0.5 ? 'medium' : 'low';
    const bgColor = congestionLevel === 'high' ? 'bg-red-50' : congestionLevel === 'medium' ? 'bg-yellow-50' : 'bg-green-50';
    const borderColor = congestionLevel === 'high' ? 'border-red-300' : congestionLevel === 'medium' ? 'border-yellow-300' : 'border-green-300';
    const textColor = congestionLevel === 'high' ? 'text-red-700' : congestionLevel === 'medium' ? 'text-yellow-700' : 'text-green-700';

    return (
        <div
            onClick={ onSelectNode }
            className={ `${ bgColor } ${ borderColor } border-2 rounded-lg p-4 cursor-pointer transition-all hover:shadow-lg ${ isSelected ? 'ring-2 ring-blue-500 shadow-lg' : ''
                }` }
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Radio className={ `w-5 h-5 ${ textColor }` } />
                    <span className="font-bold text-gray-800">{ node.nodeId }</span>
                </div>
                { congestionLevel === 'high' && <AlertTriangle className="w-5 h-5 text-red-500" /> }
            </div>

            <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                    <span className="text-gray-600">Total Packets:</span>
                    <span className="font-semibold">{ node.totalPackets || 0 }</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-gray-600">Avg Queue:</span>
                    <span className="font-semibold">{ safeFixed( node.avgQueueSize ) }</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-gray-600">Bandwidth:</span>
                    <span className={ `font-semibold ${ textColor }` }>
                        { safeFixed( node.avgBandwidthUtil * 100, 1 ) }%
                    </span>
                </div>
                <div className="flex justify-between">
                    <span className="text-gray-600">Avg Latency:</span>
                    <span className="font-semibold">{ safeFixed( node.avgLatency ) }ms</span>
                </div>
            </div>

            <div className="mt-3 pt-3 border-t border-gray-300">
                <div className="flex justify-between text-xs">
                    <span className="text-blue-600">Dijkstra: { node.algorithms.dijkstra || 0 }</span>
                    <span className="text-purple-600">RL: { node.algorithms.rl || 0 }</span>
                </div>
            </div>
        </div>
    );
};

// Component: Packet Flow Through Node
export const PacketFlowDetail = ( { node, batch }: { node: NodeCongestion; batch: NetworkBatch } ) => {
    const packetsAtNode = useMemo( () => {
        const packets: Array<{ packet: Packet; algorithm: string; pairIndex: number }> = [];

        batch.packets.forEach( ( pair, pairIdx ) => {
            [ pair.dijkstraPacket, pair.rlPacket ].forEach( packet => {  // ✅ Fixed: rlPacket (capital P)
                if (!packet) return;  // ✅ Null check
                
                // Check if this node ROUTED the packet (fromNodeId), not just received it
                // Exclude hops where toNodeId starts with "USER:" (final delivery to user)
                const hasRoutedPacket = packet.hopRecords.some(
                    hop => hop.fromNodeId === node.nodeId && !hop.toNodeId.startsWith('USER:')
                );
                if ( hasRoutedPacket ) {
                    packets.push( {
                        packet,
                        algorithm: packet.isUseRL ? 'RL' : 'Dijkstra',  // ✅ Fixed: Match backend field name
                        pairIndex: pairIdx
                    } );
                }
            } );
        } );

        return packets;
    }, [ node, batch ] );

    return (
        <div className="bg-white rounded-lg border-2 border-blue-400 p-6 w-full">
            <div className="flex items-center gap-3 mb-4">
                <Layers className="w-6 h-6 text-blue-600" />
                <div>
                    <h3 className="text-xl font-bold text-gray-800">{ node.nodeId }</h3>
                    <p className="text-sm text-gray-600">{ packetsAtNode.length } packets routed through this node</p>
                </div>
            </div>

            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                { packetsAtNode.length === 0 ? (
                    <div className="text-center text-gray-500 py-8">
                        No packets found for this node
                    </div>
                ) : (
                    packetsAtNode.map( ( item, idx ) => {
                        const hopAtNode = item.packet.hopRecords.find(
                            h => h.fromNodeId === node.nodeId
                        );

                        return (
                            <div
                                key={ idx }
                                className={ `p-3 rounded border ${ item.algorithm === 'Dijkstra'
                                    ? 'bg-blue-50 border-blue-200'
                                    : 'bg-purple-50 border-purple-200'
                                    }` }
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-mono text-gray-600">
                                        Pair #{ item.pairIndex + 1 } - { item.packet.packetId.substring( 0, 12 ) }...
                                    </span>
                                    <span className={ `text-xs font-semibold ${ item.algorithm === 'Dijkstra' ? 'text-blue-600' : 'text-purple-600'
                                        }` }>
                                        { item.algorithm }
                                    </span>
                                </div>

                                { hopAtNode && (
                                    <div className="grid grid-cols-3 gap-2 text-xs">
                                        <div>
                                            <div className="text-gray-500">Next Hop</div>
                                            <div className="font-semibold">{ hopAtNode.toNodeId }</div>
                                        </div>
                                        <div>
                                            <div className="text-gray-500">Queue Size</div>
                                            <div className="font-semibold">{ hopAtNode.fromNodeBufferState.queueSize || 0 }</div>
                                        </div>
                                        <div>
                                            <div className="text-gray-500">Bandwidth</div>
                                            <div className="font-semibold">
                                                { safeFixed( hopAtNode.fromNodeBufferState.bandwidthUtilization * 100, 0 ) }%
                                            </div>
                                        </div>
                                    </div>
                                ) }

                                <div className="mt-2 text-xs text-gray-600">
                                    Route: { item.packet.stationSource } → { item.packet.stationDest }
                                </div>
                            </div>
                        );
                    } )
                ) }
            </div>
        </div>
    );
};

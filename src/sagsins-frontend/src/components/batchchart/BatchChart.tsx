// import { useMemo } from 'react';
// import { Activity, AlertTriangle, TrendingUp, Radio, Layers } from 'lucide-react';
// import type { NetworkBatch, NodeCongestion, Packet } from '../../types/ComparisonTypes';

// // Utility function to safely format numbers
// const safeFixed = (value: number | undefined | null, decimals: number = 1): string => {
//     if (value === undefined || value === null || isNaN(value) || !isFinite(value)) {
//         return '0';
//     }
//     return value.toFixed(decimals);
// };

// // Component: Node Congestion Card
// export const NodeCongestionCard = ({ node, onSelectNode, isSelected }: {
//     node: NodeCongestion;
//     onSelectNode: () => void;
//     isSelected: boolean;
// }) => {
//     const congestionLevel = node.avgBandwidthUtil > 0.8 ? 'high' : node.avgBandwidthUtil > 0.5 ? 'medium' : 'low';
//     const bgColor = congestionLevel === 'high' ? 'bg-red-50' : congestionLevel === 'medium' ? 'bg-yellow-50' : 'bg-green-50';
//     const borderColor = congestionLevel === 'high' ? 'border-red-300' : congestionLevel === 'medium' ? 'border-yellow-300' : 'border-green-300';
//     const textColor = congestionLevel === 'high' ? 'text-red-700' : congestionLevel === 'medium' ? 'text-yellow-700' : 'text-green-700';

//     return (
//         <div
//             onClick={onSelectNode}
//             className={`${bgColor} ${borderColor} border-2 rounded-lg p-4 cursor-pointer transition-all hover:shadow-lg ${isSelected ? 'ring-2 ring-blue-500 shadow-lg' : ''
//                 }`}
//         >
//             <div className="flex items-center justify-between mb-3">
//                 <div className="flex items-center gap-2">
//                     <Radio className={`w-5 h-5 ${textColor}`} />
//                     <span className="font-bold text-gray-800">{node.nodeId}</span>
//                 </div>
//                 {congestionLevel === 'high' && <AlertTriangle className="w-5 h-5 text-red-500" />}
//             </div>

//             <div className="space-y-2 text-sm">
//                 <div className="flex justify-between">
//                     <span className="text-gray-600">Total Packets:</span>
//                     <span className="font-semibold">{node.totalPackets || 0}</span>
//                 </div>
//                 <div className="flex justify-between">
//                     <span className="text-gray-600">Avg Queue:</span>
//                     <span className="font-semibold">{safeFixed(node.avgQueueSize)}</span>
//                 </div>
//                 <div className="flex justify-between">
//                     <span className="text-gray-600">Bandwidth:</span>
//                     <span className={`font-semibold ${textColor}`}>
//                         {safeFixed(node.avgBandwidthUtil * 100, 1)}%
//                     </span>
//                 </div>
//                 <div className="flex justify-between">
//                     <span className="text-gray-600">Avg Latency:</span>
//                     <span className="font-semibold">{safeFixed(node.avgLatency)}ms</span>
//                 </div>
//             </div>

//             <div className="mt-3 pt-3 border-t border-gray-300">
//                 <div className="flex justify-between text-xs">
//                     <span className="text-blue-600">Dijkstra: {node.algorithms.dijkstra || 0}</span>
//                     <span className="text-purple-600">RL: {node.algorithms.rl || 0}</span>
//                 </div>
//             </div>
//         </div>
//     );
// };

// // Component: Packet Flow Through Node
// export const PacketFlowDetail = ({ node, batch }: { node: NodeCongestion; batch: NetworkBatch }) => {
//     const packetsAtNode = useMemo(() => {
//         const packets: Array<{ packet: Packet; algorithm: string; pairIndex: number }> = [];

//         batch.packets.forEach((pair, pairIdx) => {
//             [pair.dijkstraPacket, pair.rlpacket].forEach(packet => {
//                 const hasNode = packet.hopRecords.some(
//                     hop => hop.fromNodeId === node.nodeId || hop.toNodeId === node.nodeId
//                 );
//                 if (hasNode) {
//                     packets.push({
//                         packet,
//                         algorithm: packet.useRL ? 'RL' : 'Dijkstra',
//                         pairIndex: pairIdx
//                     });
//                 }
//             });
//         });

//         return packets;
//     }, [node, batch]);

//     return (
//         <div className="bg-white rounded-lg border-2 border-blue-400 p-6 w-full">
//             <div className="flex items-center gap-3 mb-4">
//                 <Layers className="w-6 h-6 text-blue-600" />
//                 <div>
//                     <h3 className="text-xl font-bold text-gray-800">{node.nodeId}</h3>
//                     <p className="text-sm text-gray-600">{packetsAtNode.length} packets routed through this node</p>
//                 </div>
//             </div>

//             <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
//                 {packetsAtNode.length === 0 ? (
//                     <div className="text-center text-gray-500 py-8">
//                         No packets found for this node
//                     </div>
//                 ) : (
//                     packetsAtNode.map((item, idx) => {
//                         const hopAtNode = item.packet.hopRecords.find(
//                             h => h.fromNodeId === node.nodeId
//                         );

//                         return (
//                             <div
//                                 key={idx}
//                                 className={`p-3 rounded border ${item.algorithm === 'Dijkstra'
//                                     ? 'bg-blue-50 border-blue-200'
//                                     : 'bg-purple-50 border-purple-200'
//                                     }`}
//                             >
//                                 <div className="flex items-center justify-between mb-2">
//                                     <span className="text-xs font-mono text-gray-600">
//                                         Pair #{item.pairIndex + 1} - {item.packet.packetId.substring(0, 12)}...
//                                     </span>
//                                     <span className={`text-xs font-semibold ${item.algorithm === 'Dijkstra' ? 'text-blue-600' : 'text-purple-600'
//                                         }`}>
//                                         {item.algorithm}
//                                     </span>
//                                 </div>

//                                 {hopAtNode && (
//                                     <div className="grid grid-cols-3 gap-2 text-xs">
//                                         <div>
//                                             <div className="text-gray-500">Next Hop</div>
//                                             <div className="font-semibold">{hopAtNode.toNodeId}</div>
//                                         </div>
//                                         <div>
//                                             <div className="text-gray-500">Queue Size</div>
//                                             <div className="font-semibold">{hopAtNode.fromNodeBufferState.queueSize || 0}</div>
//                                         </div>
//                                         <div>
//                                             <div className="text-gray-500">Bandwidth</div>
//                                             <div className="font-semibold">
//                                                 {safeFixed(hopAtNode.fromNodeBufferState.bandwidthUtilization * 100, 0)}%
//                                             </div>
//                                         </div>
//                                     </div>
//                                 )}

//                                 <div className="mt-2 text-xs text-gray-600">
//                                     Route: {item.packet.stationSource} → {item.packet.stationDest}
//                                 </div>
//                             </div>
//                         );
//                     })
//                 )}
//             </div>
//         </div>
//     );
// };

// Component: Network Topology View
// export const NetworkTopologyView = ({
//     congestionMap,
//     selectedNode,
//     onSelectNode
// }: {
//     congestionMap: NodeCongestion[];
//     selectedNode: string | null;
//     onSelectNode: (nodeId: string) => void;
// }) => {
//     return (
//         <div className="bg-white rounded-lg border border-gray-300 p-6 w-full">
//             <div className="flex items-center gap-2 mb-4">
//                 <Activity className="w-5 h-5 text-gray-700" />
//                 <h3 className="text-lg font-bold text-gray-800">Network Nodes Congestion</h3>
//             </div>

//             {congestionMap.length === 0 ? (
//                 <div className="text-center text-gray-500 py-8">
//                     No nodes available
//                 </div>
//             ) : (
//                 <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
//                     {congestionMap.map(node => (
//                         <NodeCongestionCard
//                             key={node.nodeId}
//                             node={node}
//                             onSelectNode={() => onSelectNode(node.nodeId)}
//                             isSelected={selectedNode === node.nodeId}
//                         />
//                     ))}
//                 </div>
//             )}
//         </div>
//     );
// };

// // Component: Batch Statistics
// export const BatchStatistics = ({ batch, congestionMap }: { batch: NetworkBatch; congestionMap: NodeCongestion[] }) => {
//     const stats = useMemo(() => {
//         const totalNodes = congestionMap.length;
//         const highCongestion = congestionMap.filter(n => n.avgBandwidthUtil > 0.8).length;
//         const mediumCongestion = congestionMap.filter(n => n.avgBandwidthUtil > 0.5 && n.avgBandwidthUtil <= 0.8).length;

//         let dijkstraTotal = 0, rlTotal = 0;
//         let validPackets = 0;

//         batch.packets.forEach(pair => {
//             const dijkstraLatency = pair.dijkstraPacket?.analysisData?.totalLatencyMs;
//             const rlLatency = pair.rlpacket?.analysisData?.totalLatencyMs;

//             if (dijkstraLatency !== undefined && !isNaN(dijkstraLatency)) {
//                 dijkstraTotal += dijkstraLatency;
//                 validPackets++;
//             }
//             if (rlLatency !== undefined && !isNaN(rlLatency)) {
//                 rlTotal += rlLatency;
//             }
//         });

//         const avgDijkstra = validPackets > 0 ? dijkstraTotal / validPackets : 0;
//         const avgRL = validPackets > 0 ? rlTotal / validPackets : 0;
//         const improvement = avgDijkstra > 0 ? ((dijkstraTotal - rlTotal) / dijkstraTotal * 100) : 0;

//         return {
//             totalNodes,
//             highCongestion,
//             mediumCongestion,
//             avgDijkstra,
//             avgRL,
//             improvement
//         };
//     }, [batch, congestionMap]);

//     return (
//         <div className="bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 rounded-lg p-6 border border-gray-300 w-full">
//             <h2 className="text-xl font-bold text-gray-800 mb-4">Batch: {batch.batchId}</h2>

//             <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
//                 <div className="bg-white rounded-lg p-4 border border-gray-200">
//                     <div className="text-xs text-gray-500 mb-1">Total Nodes</div>
//                     <div className="text-2xl font-bold text-gray-800">{stats.totalNodes}</div>
//                 </div>

//                 <div className="bg-white rounded-lg p-4 border border-red-200">
//                     <div className="text-xs text-gray-500 mb-1">High Congestion</div>
//                     <div className="text-2xl font-bold text-red-600">{stats.highCongestion}</div>
//                 </div>

//                 <div className="bg-white rounded-lg p-4 border border-yellow-200">
//                     <div className="text-xs text-gray-500 mb-1">Medium Congestion</div>
//                     <div className="text-2xl font-bold text-yellow-600">{stats.mediumCongestion}</div>
//                 </div>

//                 <div className="bg-white rounded-lg p-4 border border-blue-200">
//                     <div className="text-xs text-gray-500 mb-1">Dijkstra Avg</div>
//                     <div className="text-2xl font-bold text-blue-600">
//                         {safeFixed(stats.avgDijkstra, 1)}<span className="text-sm">ms</span>
//                     </div>
//                 </div>

//                 <div className="bg-white rounded-lg p-4 border border-purple-200">
//                     <div className="text-xs text-gray-500 mb-1">RL Avg</div>
//                     <div className="text-2xl font-bold text-purple-600">
//                         {safeFixed(stats.avgRL, 1)}<span className="text-sm">ms</span>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// // Component: Algorithm Comparison Summary
// export const AlgorithmComparisonChart = ({ batch }: { batch: NetworkBatch }) => {
//     const comparison = useMemo(() => {
//         let dijkstraWins = 0, rlWins = 0, ties = 0;
//         let dijkstraDropped = 0, rlDropped = 0;

//         batch.packets.forEach(pair => {
//             const dijkstraLatency = pair.dijkstraPacket?.analysisData?.totalLatencyMs;
//             const rlLatency = pair.rlpacket?.analysisData?.totalLatencyMs;

//             if (dijkstraLatency !== undefined && rlLatency !== undefined && 
//                 !isNaN(dijkstraLatency) && !isNaN(rlLatency)) {
//                 const diff = dijkstraLatency - rlLatency;
//                 if (Math.abs(diff) < 1) ties++;
//                 else if (diff > 0) rlWins++;
//                 else dijkstraWins++;
//             }

//             if (pair.dijkstraPacket?.dropped) dijkstraDropped++;
//             if (pair.rlpacket?.dropped) rlDropped++;
//         });

//         return { dijkstraWins, rlWins, ties, dijkstraDropped, rlDropped };
//     }, [batch]);

//     const totalPairs = batch.totalPairPackets || 1; // Avoid division by zero

//     return (
//         <div className="bg-white rounded-lg border border-gray-300 p-6 w-full">
//             <div className="flex items-center gap-2 mb-4">
//                 <TrendingUp className="w-5 h-5 text-gray-700" />
//                 <h3 className="text-lg font-bold text-gray-800">Algorithm Performance</h3>
//             </div>

//             <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
//                 <div>
//                     <div className="text-sm text-gray-600 mb-3">Win Rate</div>
//                     <div className="space-y-2">
//                         <div className="flex items-center gap-3">
//                             <div className="w-24 text-sm text-blue-600 font-medium">Dijkstra</div>
//                             <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
//                                 <div
//                                     className="bg-blue-500 h-full flex items-center justify-end pr-2"
//                                     style={{ width: `${(comparison.dijkstraWins / totalPairs) * 100}%` }}
//                                 >
//                                     {comparison.dijkstraWins > 0 && (
//                                         <span className="text-xs text-white font-bold">{comparison.dijkstraWins}</span>
//                                     )}
//                                 </div>
//                             </div>
//                             <span className="text-xs text-gray-500 w-12 text-right">
//                                 {safeFixed((comparison.dijkstraWins / totalPairs) * 100, 0)}%
//                             </span>
//                         </div>
//                         <div className="flex items-center gap-3">
//                             <div className="w-24 text-sm text-purple-600 font-medium">RL</div>
//                             <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
//                                 <div
//                                     className="bg-purple-500 h-full flex items-center justify-end pr-2"
//                                     style={{ width: `${(comparison.rlWins / totalPairs) * 100}%` }}
//                                 >
//                                     {comparison.rlWins > 0 && (
//                                         <span className="text-xs text-white font-bold">{comparison.rlWins}</span>
//                                     )}
//                                 </div>
//                             </div>
//                             <span className="text-xs text-gray-500 w-12 text-right">
//                                 {safeFixed((comparison.rlWins / totalPairs) * 100, 0)}%
//                             </span>
//                         </div>
//                     </div>
//                 </div>

//                 <div>
//                     <div className="text-sm text-gray-600 mb-3">Dropped Packets</div>
//                     <div className="space-y-3">
//                         <div className="flex justify-between items-center p-3 bg-blue-50 rounded">
//                             <span className="text-sm text-blue-700">Dijkstra</span>
//                             <span className="text-xl font-bold text-blue-700">{comparison.dijkstraDropped}</span>
//                         </div>
//                         <div className="flex justify-between items-center p-3 bg-purple-50 rounded">
//                             <span className="text-sm text-purple-700">RL</span>
//                             <span className="text-xl font-bold text-purple-700">{comparison.rlDropped}</span>
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };
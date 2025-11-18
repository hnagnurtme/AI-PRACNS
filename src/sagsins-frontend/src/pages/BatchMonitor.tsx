import React, { useState, useMemo } from 'react';
import { useBatchWebSocket } from '../hooks/useBatchWebSocket'; // Giả định hook của bạn
import { calculateCongestionMap } from '../utils/calculateCongestionMap';
import { BatchStatistics } from '../components/batchchart/BatchStatistics';
import { NetworkTopologyView } from '../components/batchchart/NetworkTopologyView';
import { PacketFlowDetail } from '../components/batchchart/PacketFlowDetail';
import { AlgorithmComparisonChart } from '../components/batchchart/AlgorithmComparisonChart';
import { ScenarioSelector } from '../components/simulation/ScenarioSelector';
import { PacketRouteGraph } from '../components/chart/PacketRouteGraph';

const DASHBOARD_ENDPOINT = import.meta.env.VITE_WS_URL; // Sử dụng biến môi trường

const BatchDashboard: React.FC = () => {
    // 1. Lấy dữ liệu lô gói tin từ WebSocket
    const { receivedBatches, connectionStatus } = useBatchWebSocket(DASHBOARD_ENDPOINT);
    
    // Lấy lô gói tin MỚI NHẤT
    const latestBatch = receivedBatches[receivedBatches.length - 1];

    // Trạng thái Node được chọn
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

    // Trạng thái Packet Pair được chọn để hiển thị route visualization
    const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0);

    // 2. TÍNH TOÁN MAP TẮC NGHẼN (Sử dụng useMemo)
    const congestionMap = useMemo(() => {
        // Tính toán tổng hợp từ TẤT CẢ các lô đã nhận nếu cần tổng lũy
        // Hoặc chỉ từ latestBatch nếu bạn chỉ quan tâm đến lô mới nhất
        if (!latestBatch) return [];
        return calculateCongestionMap([latestBatch]); // Chỉ tính cho lô mới nhất
    }, [latestBatch]); // Chạy lại khi lô mới nhất thay đổi

    // Dữ liệu Node được chọn
    const selectedNodeData = useMemo(() => {
        return congestionMap.find(node => node.nodeId === selectedNodeId) || null;
    }, [congestionMap, selectedNodeId]);

    if (connectionStatus !== 'CONNECTED' || !latestBatch) {
        return (
            <div className="p-10 text-center text-gray-500">
                Kết nối: {connectionStatus}. Đang chờ dữ liệu lô gói tin đầu tiên...
            </div>
        );
    }

    return (
        <div className="space-y-6 p-6 max-w-7xl mx-auto">

            {/* Scenario Selector for Network Simulation */}
            <ScenarioSelector />

            {/* 1. Thống kê Tổng quan Batch */}
            <BatchStatistics batch={latestBatch} congestionMap={congestionMap} />

            {/* 2. So sánh Thuật toán & Tắc nghẽn Node */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                    <NetworkTopologyView
                        congestionMap={congestionMap}
                        selectedNode={selectedNodeId}
                        onSelectNode={setSelectedNodeId}
                    />
                </div>
                {/* Chi tiết gói tin (Chỉ hiển thị khi có Node được chọn) */}
                {selectedNodeData && (
                    <PacketFlowDetail 
                        node={selectedNodeData} 
                        batch={latestBatch} 
                    />
                )}
            </div>

            {/* 3. Biểu đồ So sánh Thuật toán */}
            <AlgorithmComparisonChart batch={latestBatch} />

            {/* 4. Route Visualization & Hop Performance */}
            {latestBatch && latestBatch.packets && latestBatch.packets.length > 0 && (
                <div className="space-y-4">
                    {/* Packet Pair Selector */}
                    {latestBatch.packets.length > 1 && (
                        <div className="bg-white rounded-lg border border-gray-300 p-4">
                            <label htmlFor="pair-selector" className="block text-sm font-medium text-gray-700 mb-2">
                                Select Packet Pair to Visualize Route
                            </label>
                            <select
                                id="pair-selector"
                                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                                value={selectedPairIndex}
                                onChange={(e) => setSelectedPairIndex(Number(e.target.value))}
                            >
                                {latestBatch.packets.map((pair, idx) => (
                                    <option key={idx} value={idx}>
                                        Pair #{idx + 1} - {pair.dijkstraPacket?.stationSource || 'N/A'} → {pair.dijkstraPacket?.stationDest || 'N/A'}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}

                    {/* Route Visualization for Selected Pair */}
                    <PacketRouteGraph data={latestBatch.packets[selectedPairIndex]} />
                </div>
            )}

        </div>
    );
};

export default BatchDashboard;
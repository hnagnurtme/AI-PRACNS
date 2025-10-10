import React, { useEffect, useState, useRef } from 'react';
import CesiumViewer from '../map/CesiumViewer'; 
import Sidebar from '../components/Sidebar'; 
import NodeDetailCard from '../components/nodes/NodeDetailCard'; 
import { useNodeStore } from '../state/nodeStore'; 
import { useNodes } from '../hooks/useNodes'; 

const Dashboard: React.FC = () => {
    // 1. Lấy trạng thái và actions từ store
    const { nodes, selectedNode } = useNodeStore();
    const { refetchNodes } = useNodes();

    // 2. Auto-refresh state
    const [autoRefresh, setAutoRefresh] = useState(true); // Mặc định bật auto-refresh
    const [refreshInterval, setRefreshInterval] = useState(5); // Refresh mỗi 5 giây
    const intervalRef = useRef<number | null>(null);

    // 3. Logic Fetch dữ liệu (Chạy khi component được mount)
    useEffect(() => {
        refetchNodes().catch(error => {
            console.error("Failed to load Nodes data from API:", error);
            // TODO: Xử lý hiển thị thông báo lỗi trên UI
        });
    }, [refetchNodes]);

    // 4. Auto-refresh logic
    useEffect(() => {
        if (autoRefresh) {
            intervalRef.current = setInterval(() => {
                refetchNodes().catch(error => {
                    console.error("Auto-refresh failed:", error);
                });
            }, refreshInterval * 1000);
        } else {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }

        // Cleanup interval on component unmount or when autoRefresh changes
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [autoRefresh, refreshInterval, refetchNodes]);

    // 5. Manual refresh handler
    const handleManualRefresh = () => {
        refetchNodes().catch(error => {
            console.error("Manual refresh failed:", error);
        });
    };

    return (
        // Sử dụng Tailwind CSS để chia layout (Map chiếm phần lớn, Sidebar cố định)
        <div className="flex w-screen h-screen overflow-hidden bg-gray-100">
            
            {/* Sidebar (Cố định chiều rộng: w-80) */}
            {/* Sidebar nhận nodes để hiển thị danh sách */}
            <Sidebar nodes={nodes} onRefresh={refetchNodes} /> 

            {/* Khu vực Map (flex-grow: chiếm hết không gian còn lại) */}
            <div className="relative flex-grow">
                
                {/* Auto-refresh Controls */}
                <div className="absolute top-4 left-4 z-20 bg-white p-3 rounded-lg shadow-lg border">
                    <div className="flex items-center space-x-3 text-sm">
                        <label className="flex items-center space-x-2">
                            <input 
                                type="checkbox" 
                                checked={autoRefresh}
                                onChange={(e) => setAutoRefresh(e.target.checked)}
                                className="w-4 h-4"
                            />
                            <span className="font-medium">Auto Refresh</span>
                        </label>
                        
                        <select 
                            value={refreshInterval}
                            onChange={(e) => setRefreshInterval(Number(e.target.value))}
                            disabled={!autoRefresh}
                            className="px-2 py-1 border rounded text-sm disabled:bg-gray-100"
                        >
                            <option value={2}>2s</option>
                            <option value={5}>5s</option>
                            <option value={10}>10s</option>
                            <option value={30}>30s</option>
                        </select>

                        <button
                            onClick={handleManualRefresh}
                            className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition"
                        >
                            🔄 Refresh Now
                        </button>
                        
                        <span className="text-gray-500 text-xs">
                            Nodes: {nodes.length}
                        </span>
                    </div>
                </div>

                {/* Cesium Viewer */}
                {/* CesiumViewer nhận nodes để render các entities */}
                <CesiumViewer nodes={nodes} />

                {/* Card thông tin chi tiết (Hiển thị nổi trên Map) */}
                {/* Card chỉ hiển thị nếu selectedNode có dữ liệu */}
                {selectedNode && (
                    <div className="absolute top-20 right-4 z-10 w-96">
                        <NodeDetailCard node={selectedNode} onRefresh={refetchNodes} />
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
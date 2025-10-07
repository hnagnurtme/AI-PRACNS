import React, { useEffect } from 'react';
import CesiumViewer from '../map/CesiumViewer'; 
import Sidebar from '../components/Sidebar'; 
import NodeDetailCard from '../components/nodes/NodeDetailCard'; 
import { useNodeStore } from '../state/nodeStore'; 
import { getAllNodes } from '../services/nodeService'; 

const Dashboard: React.FC = () => {
    // 1. Lấy trạng thái và actions từ store
    const { nodes, selectedNode, setNodes } = useNodeStore();

    // 2. Logic Fetch dữ liệu (Chạy khi component được mount)
    useEffect(() => {
        const fetchNodes = async () => {
            // Có thể đặt trạng thái isLoading ở đây
            try {
                const fetchedNodes = await getAllNodes();
                setNodes(fetchedNodes); // Cập nhật danh sách nodes vào store
            } catch (error) {
                console.error("Failed to load Nodes data from API:", error);
                // TODO: Xử lý hiển thị thông báo lỗi trên UI
            }
            // Tắt trạng thái isLoading ở đây
        };
        fetchNodes();
    }, [setNodes]); // Dependency [setNodes] đảm bảo hàm chỉ chạy khi setNodes thay đổi

    return (
        // Sử dụng Tailwind CSS để chia layout (Map chiếm phần lớn, Sidebar cố định)
        <div className="flex w-screen h-screen overflow-hidden bg-gray-100">
            
            {/* Sidebar (Cố định chiều rộng: w-80) */}
            {/* Sidebar nhận nodes để hiển thị danh sách */}
            <Sidebar nodes={nodes} /> 

            {/* Khu vực Map (flex-grow: chiếm hết không gian còn lại) */}
            <div className="relative flex-grow">
                
                {/* Cesium Viewer */}
                {/* CesiumViewer nhận nodes để render các entities */}
                <CesiumViewer nodes={nodes} />

                {/* Card thông tin chi tiết (Hiển thị nổi trên Map) */}
                {/* Card chỉ hiển thị nếu selectedNode có dữ liệu */}
                {selectedNode && (
                    <div className="absolute top-4 right-4 z-10 w-96">
                        <NodeDetailCard node={selectedNode} />
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
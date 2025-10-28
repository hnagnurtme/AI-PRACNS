// src/Dashboard.tsx
import React, { useEffect } from 'react';
import CesiumViewer from '../map/CesiumViewer';
import Sidebar from '../components/Sidebar';
import NodeDetailCard from '../components/nodes/NodeDetailCard';
import { useNodeStore } from '../state/nodeStore';
import { useNodes } from '../hooks/useNodes';

const Dashboard: React.FC = () => {
    const { nodes, selectedNode } = useNodeStore();
    const { refetchNodes } = useNodes();

    useEffect( () => {
        refetchNodes().catch( ( error ) =>
            console.error( 'Failed to load Nodes data from API:', error )
        );
    }, [ refetchNodes ] );

    return (
        // SỬA 1: Khóa layout vào 100% màn hình và CẤM TOÀN BỘ TRANG CUỘN
        <div className="flex h-screen w-screen overflow-hidden bg-gray-100">
            
            {/* SỬA 2: Bọc Sidebar trong một div cố định chiều rộng
                 và BẮT BUỘC NÓ TỰ CUỘN NỘI BỘ */}
            <div className="h-full flex-shrink-0 overflow-y-auto bg-white shadow-lg">
                {/* Giả sử w-80 (320px) là độ rộng sidebar bạn muốn */}
                <Sidebar nodes={ nodes } />
            </div>


            {/* SỬA 3: Khu vực bản đồ giờ sẽ lấp đầy không gian còn lại
                 và cũng bị cấm cuộn (overflow-hidden) */}
            <div className="relative flex-grow h-full overflow-hidden">
                <CesiumViewer nodes={ nodes } />

                { selectedNode && (
                    <NodeDetailCard node={ selectedNode } />
                ) }
            </div>
        </div>
    );
};

export default Dashboard;
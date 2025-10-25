// src/components/Sidebar.tsx
import React, { useState } from 'react'; // [SỬA 1] Thêm 'useState'
import { useNodeStore } from '../state/nodeStore';
import type { NodeDTO } from '../types/NodeTypes';

interface SidebarProps {
    nodes: NodeDTO[];
}

const Sidebar: React.FC<SidebarProps> = ({ nodes }) => {
    // [SỬA 1] Thêm state để quản lý việc ẩn/hiện sidebar
    const [isOpen, setIsOpen] = useState(true);

    // Lấy 'selectAndFly' từ store (đã sửa ở lần trước)
    const { selectAndFly, selectedNode } = useNodeStore();


    const handleNodeClick = (node: NodeDTO) => {
        selectAndFly(node); 
    };

    return (
        // [SỬA 1] Thêm 'transition-all' và thay đổi 'width' dựa trên state 'isOpen'
        <div 
            className={`relative z-10 bg-gray-900 text-white flex flex-col h-full shadow-2xl
                        transition-all duration-300 ease-in-out ${isOpen ? 'w-80' : 'w-16'}`}
        >
            {/* Header và nút Thu/Gọn */}
            <div className={`p-4 flex items-center border-b border-gray-700 ${isOpen ? 'justify-between' : 'justify-center'}`}>
                {/* [SỬA 1] Ẩn tiêu đề khi sidebar thu gọn */}
                <h2 className={`text-xl font-bold truncate ${!isOpen && 'hidden'}`}>
                    Nodes ({nodes.length})
                </h2>

                {/* [SỬA 1] Nút để thu/gọn sidebar */}
                <button 
                    onClick={() => setIsOpen(!isOpen)} 
                    className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700"
                    title={isOpen ? "Thu gọn" : "Mở rộng"}
                >
                    {/* Dùng ký tự » (thu) và « (mở) */}
                    {isOpen ? '«' : '»'}
                </button>
            </div>

            {/* Danh sách Nodes */}
            <div className="flex-grow overflow-y-auto">
                {nodes.length === 0 ? (
                    // [SỬA 1] Ẩn text khi sidebar thu gọn
                    <p className={`p-4 text-gray-500 italic ${!isOpen && 'hidden'}`}>
                        No nodes deployed.
                    </p>
                ) : (
                    nodes.map((node) => (
                        <div
                            key={node.nodeId}
                            onClick={() => handleNodeClick(node)}
                            // [SỬA 1] Thêm 'title' để hiển thị tên khi thu gọn và căn giữa
                            title={!isOpen ? node.nodeName : undefined}
                            className={`p-3 border-b border-gray-800 cursor-pointer transition duration-150 
                                        flex items-center gap-3 ${!isOpen && 'justify-center'}
                                        ${selectedNode?.nodeId === node.nodeId ? 'bg-indigo-600 font-semibold' : 'hover:bg-gray-700'}`}
                        >
                            {/* [SỬA 3] Đèn trạng thái dựa trên 'node.healthy' */}
                            <span 
                                className={`w-3 h-3 rounded-full flex-shrink-0 ${node.healthy ? 'bg-green-400' : 'bg-red-500'}`}
                                title={node.healthy ? 'Healthy' : 'Unhealthy'}
                            ></span>

                            {/* [SỬA 2] Hiển thị 'nodeName' và ẩn khi thu gọn */}
                            <span className={`truncate ${!isOpen && 'hidden'}`}>
                                {node.nodeName}
                            </span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default Sidebar;
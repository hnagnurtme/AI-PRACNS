// src/components/Sidebar.tsx
import React, { useState } from 'react';
import { useNodeStore } from '../state/nodeStore';
import type { NodeDTO } from '../types/NodeTypes';

interface SidebarProps {
    nodes: NodeDTO[];
}

const Sidebar: React.FC<SidebarProps> = ({ nodes }) => {
    const [isOpen, setIsOpen] = useState(false);
    const { selectAndFly, selectedNode } = useNodeStore();

    const handleNodeClick = (node: NodeDTO) => {
        selectAndFly(node); 
    };

    return (
        <div 
            className={`relative z-10 bg-gray-900 text-white flex flex-col h-full shadow-2xl
                        transition-all duration-300 ease-in-out ${isOpen ? 'w-80' : 'w-16'}`}
        >
            {/* Header và nút Thu/Gọn */}
            <div className={`p-4 flex items-center border-b border-gray-700 ${isOpen ? 'justify-between' : 'justify-center'}`}>
                <h2 className={`text-xl font-bold truncate ${!isOpen && 'hidden'}`}>
                    Nodes ({nodes.length})
                </h2>
                <button 
                    onClick={() => setIsOpen(!isOpen)} 
                    className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700"
                    title={isOpen ? "Thu gọn" : "Mở rộng"}
                >
                    {isOpen ? '«' : '»'}
                </button>
            </div>

            {/* [SỬA] Thêm 'overflow-x-visible' vào đây */}
            <div className="flex-grow overflow-y-auto overflow-x-visible">
                {nodes.length === 0 ? (
                    <p className={`p-4 text-gray-500 italic ${!isOpen && 'hidden'}`}>
                        No nodes deployed.
                    </p>
                ) : (
                    nodes.map((node) => (
                        <div
                            key={node.nodeId}
                            onClick={() => handleNodeClick(node)}
                            // Thêm 'group' và 'relative'
                            className={`group relative p-3 border-b border-gray-800 cursor-pointer transition duration-150 
                                        flex items-center gap-3 ${!isOpen && 'justify-center'}
                                        ${selectedNode?.nodeId === node.nodeId ? 'bg-indigo-600 font-semibold' : 'hover:bg-gray-700'}`}
                        >
                            {/* Đèn trạng thái */}
                            <span 
                                className={`w-3 h-3 rounded-full flex-shrink-0 ${node.healthy ? 'bg-green-400' : 'bg-red-500'}`}
                                title={node.healthy ? 'Healthy' : 'Unhealthy'}
                            ></span>

                            {/* Tên node (ẩn khi thu gọn) */}
                            <span className={`truncate ${!isOpen && 'hidden'}`}>
                                {node.nodeName}
                            </span>

                            {/* Tooltip CSS (Chỉ render khi đóng) */}
                            {!isOpen && (
                                <span className="absolute left-full ml-2 px-3 py-1.5 bg-gray-700 text-white text-sm rounded-md shadow-lg
                                                 invisible opacity-0 group-hover:visible group-hover:opacity-100 transition-all
                                                 whitespace-nowrap z-50">
                                    {node.nodeName}
                                </span>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default Sidebar;
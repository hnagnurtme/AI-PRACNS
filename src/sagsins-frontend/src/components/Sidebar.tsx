import React from 'react';

import { useNodeStore } from '../state/nodeStore';
import type { NodeDTO } from '../types/NodeTypes';

/**
 * Props cho Sidebar. Nhận nodes từ Dashboard.
 */
interface SidebarProps {
    nodes: NodeDTO[];
}

const Sidebar: React.FC<SidebarProps> = ({ nodes }) => {
    const { setSelectedNode, selectedNode, runningNodes } = useNodeStore();

    // Hàm xử lý khi người dùng bấm vào một Node trong danh sách
    const handleNodeClick = (node: NodeDTO) => {
        // Cập nhật state toàn cục, CesiumViewer sẽ phản ứng lại
        setSelectedNode(node); 
    };

    return (
        <div className="w-80 bg-gray-900 text-white flex flex-col h-full shadow-2xl">
            {/* Header và nút Thêm Node */}
            <div className="p-4 flex justify-between items-center border-b border-gray-700">
                <h2 className="text-xl font-bold">SAGSINs Nodes ({nodes.length})</h2>
            </div>

            {/* Danh sách Nodes */}
            <div className="flex-grow overflow-y-auto">
                {nodes.length === 0 ? (
                    <p className="p-4 text-gray-500 italic">No nodes deployed.</p>
                ) : (
                    nodes.map((node) => (
                        <div
                            key={node.nodeId}
                            onClick={() => handleNodeClick(node)}
                            className={`p-3 border-b border-gray-800 cursor-pointer transition duration-150 
                                ${selectedNode?.nodeId === node.nodeId ? 'bg-indigo-600 font-semibold' : 'hover:bg-gray-700'}`}
                        >
                            <div className="flex justify-between items-center">
                                <span className="truncate">{node.nodeType}: {node.nodeId.substring(0, 8)}...</span>
                                <span 
                                    className={`w-3 h-3 rounded-full ${runningNodes.has(node.nodeId) ? 'bg-green-400' : 'bg-red-500'}`}
                                    title={runningNodes.has(node.nodeId) ? 'Running' : 'Stopped'}
                                ></span>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default Sidebar;
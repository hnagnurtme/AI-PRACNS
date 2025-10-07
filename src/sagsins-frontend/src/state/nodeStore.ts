import { create } from 'zustand';
import type { NodeDTO } from '../types/NodeTypes';


interface NodeState {
    nodes: NodeDTO[];
    selectedNode: NodeDTO | null;
    
    // Actions
    setNodes: (nodes: NodeDTO[]) => void;
    setSelectedNode: (node: NodeDTO | null) => void;
    
    // Action helper để cập nhật một node cụ thể trong danh sách
    updateNodeInStore: (updatedNode: NodeDTO) => void;
    removeNodeFromStore: (nodeId: string) => void;
}

export const useNodeStore = create<NodeState>((set) => ({
    nodes: [],
    selectedNode: null,
    
    setNodes: (nodes) => set({ nodes }),
    
    setSelectedNode: (node) => set({ 
        selectedNode: node 
    }),

    updateNodeInStore: (updatedNode) => 
        set((state) => ({
            nodes: state.nodes.map(node => 
                node.nodeId === updatedNode.nodeId ? updatedNode : node
            ),
            // Cập nhật selectedNode nếu nó là Node đang được chọn
            selectedNode: state.selectedNode && state.selectedNode.nodeId === updatedNode.nodeId 
                ? updatedNode : state.selectedNode
        })),

    removeNodeFromStore: (nodeId) => 
        set((state) => ({
            nodes: state.nodes.filter(node => node.nodeId !== nodeId),
            // Đặt selectedNode về null nếu Node bị xóa là Node đang được chọn
            selectedNode: state.selectedNode && state.selectedNode.nodeId === nodeId 
                ? null : state.selectedNode
        }))
}));
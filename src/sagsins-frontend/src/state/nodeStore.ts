import { create } from 'zustand';
import type { NodeDTO } from '../types/NodeTypes';


interface NodeState {
    nodes: NodeDTO[];
    selectedNode: NodeDTO | null;
    cameraFollowMode: boolean;
    runningNodes: Set<string>; // Track which nodes are currently running
    
    // Actions
    setNodes: (nodes: NodeDTO[]) => void;
    setSelectedNode: (node: NodeDTO | null) => void;
    setCameraFollowMode: (follow: boolean) => void;
    setNodeRunning: (nodeId: string, isRunning: boolean) => void;
    
    // Action helper để cập nhật một node cụ thể trong danh sách
    updateNodeInStore: (updatedNode: NodeDTO) => void;
    removeNodeFromStore: (nodeId: string) => void;
}

export const useNodeStore = create<NodeState>((set) => ({
    nodes: [],
    selectedNode: null,
    cameraFollowMode: false,
    runningNodes: new Set<string>(),
    
    setNodes: (nodes) => set({ nodes }),
    
    setSelectedNode: (node) => set({ 
        selectedNode: node 
    }),

    setCameraFollowMode: (follow) => set({ cameraFollowMode: follow }),

    setNodeRunning: (nodeId, isRunning) => 
        set((state) => {
            const newRunningNodes = new Set(state.runningNodes);
            if (isRunning) {
                newRunningNodes.add(nodeId);
            } else {
                newRunningNodes.delete(nodeId);
            }
            return { runningNodes: newRunningNodes };
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
        set((state) => {
            const newRunningNodes = new Set(state.runningNodes);
            newRunningNodes.delete(nodeId); // Remove from running nodes too
            
            return {
                nodes: state.nodes.filter(node => node.nodeId !== nodeId),
                runningNodes: newRunningNodes,
                // Đặt selectedNode về null nếu Node bị xóa là Node đang được chọn
                selectedNode: state.selectedNode && state.selectedNode.nodeId === nodeId 
                    ? null : state.selectedNode
            };
        })
}));
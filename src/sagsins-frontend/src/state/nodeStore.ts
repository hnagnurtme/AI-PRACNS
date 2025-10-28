import { create } from 'zustand';
import type { NodeDTO } from '../types/NodeTypes';

interface NodeState {
    nodes: NodeDTO[];
    selectedNode: NodeDTO | null;
    cameraFollowMode: boolean;
    runningNodes: Set<string>;
    flyToTrigger: number; 
    
    setNodes: (nodes: NodeDTO[]) => void;
    setSelectedNode: (node: NodeDTO | null) => void;
    selectAndFly: (node: NodeDTO) => void; 
    setCameraFollowMode: (follow: boolean) => void;
    setNodeRunning: (nodeId: string, isRunning: boolean) => void;
    
    // Action helper
    updateNodeInStore: (updatedNode: NodeDTO) => void;
    removeNodeFromStore: (nodeId: string) => void;
}

export const useNodeStore = create<NodeState>((set) => ({
    nodes: [],
    selectedNode: null,
    cameraFollowMode: false,
    runningNodes: new Set<string>(),
    flyToTrigger: 0, // <-- 3. Thêm giá trị ban đầu
    
    setNodes: (nodes) => set({ nodes }),
    
    // [SỬA] Action này CHỈ CHỌN (dùng cho click map)
    setSelectedNode: (node) => set({ 
        selectedNode: node,
        cameraFollowMode: false // Tự động tắt follow khi chọn node mới
    }),

    // [MỚI] Action này CHỌN VÀ KÍCH HOẠT BAY (dùng cho sidebar)
    selectAndFly: (node) => set((state) => ({
        selectedNode: node,
        cameraFollowMode: false, // Tắt follow
        flyToTrigger: state.flyToTrigger + 1 // <-- 4. Tăng cò súng
    })),

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
            selectedNode: state.selectedNode && state.selectedNode.nodeId === updatedNode.nodeId 
                ? updatedNode : state.selectedNode
        })),

    removeNodeFromStore: (nodeId) => 
        set((state) => {
            const newRunningNodes = new Set(state.runningNodes);
            newRunningNodes.delete(nodeId); 
            
            return {
                nodes: state.nodes.filter(node => node.nodeId !== nodeId),
                runningNodes: newRunningNodes,
                selectedNode: state.selectedNode && state.selectedNode.nodeId === nodeId 
                    ? null : state.selectedNode
            };
        })
}));
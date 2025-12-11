import { create } from 'zustand';
import type { NodeDTO } from '../types/NodeTypes';

interface NodeState {
    nodes: NodeDTO[];
    selectedNode: NodeDTO | null;
    cameraFollowMode: boolean;
    runningNodes: Set<string>;
    flyToTrigger: number;
    recentlyUpdatedNodes: Set<string>; // Track nodes with recent resource changes
    
    setNodes: (nodes: NodeDTO[]) => void;
    setSelectedNode: (node: NodeDTO | null) => void;
    selectAndFly: (node: NodeDTO) => void; 
    setCameraFollowMode: (follow: boolean) => void;
    setNodeRunning: (nodeId: string, isRunning: boolean) => void;
    
    updateNodeInStore: (updatedNode: NodeDTO) => void;
    removeNodeFromStore: (nodeId: string) => void;
}

export const useNodeStore = create<NodeState>((set) => ({
    nodes: [],
    selectedNode: null,
    cameraFollowMode: false,
    runningNodes: new Set<string>(),
    flyToTrigger: 0,
    recentlyUpdatedNodes: new Set<string>(), 
    
    setNodes: (nodes) => set({ nodes }),
    
    setSelectedNode: (node) => set({ 
        selectedNode: node,
        cameraFollowMode: false 
    }),

    selectAndFly: (node) => set((state) => ({
        selectedNode: node,
        cameraFollowMode: false, 
        flyToTrigger: state.flyToTrigger + 1 
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
        set((state) => {
            const existingNode = state.nodes.find(n => n.nodeId === updatedNode.nodeId);
            const hasResourceChange = existingNode && (
                existingNode.resourceUtilization !== updatedNode.resourceUtilization ||
                existingNode.batteryChargePercent !== updatedNode.batteryChargePercent ||
                existingNode.currentPacketCount !== updatedNode.currentPacketCount
            );
            
            const newRecentlyUpdated = new Set(state.recentlyUpdatedNodes);
            if (hasResourceChange) {
                newRecentlyUpdated.add(updatedNode.nodeId);
                // Auto-remove after 2 seconds
                setTimeout(() => {
                    set((s) => {
                        const updated = new Set(s.recentlyUpdatedNodes);
                        updated.delete(updatedNode.nodeId);
                        return { recentlyUpdatedNodes: updated };
                    });
                }, 2000);
            }
            
            return {
                nodes: state.nodes.map(node => 
                    node.nodeId === updatedNode.nodeId ? updatedNode : node
                ),
                selectedNode: state.selectedNode && state.selectedNode.nodeId === updatedNode.nodeId 
                    ? updatedNode : state.selectedNode,
                recentlyUpdatedNodes: newRecentlyUpdated
            };
        }),

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
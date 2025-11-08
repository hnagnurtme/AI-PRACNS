import { useCallback, useEffect } from 'react';
import { useNodeStore } from '../state/nodeStore';
import { getAllNodes } from '../services/nodeService';
import { useNodeStatusWebSocket } from './useNodeStatusWebSocket';

export const useNodes = () => {
    const { setNodes, updateNodeInStore } = useNodeStore();

    // Setup WebSocket for real-time node updates
    useNodeStatusWebSocket({
        url: import.meta.env.VITE_WS_URL || 'http://localhost:8080/ws',
        onNodeUpdate: (updatedNode) => {
            console.log('🔄 Received node update via WebSocket:', updatedNode);
            updateNodeInStore(updatedNode);
        }
    });

    const refetchNodes = useCallback(async () => {
        try {
            const fetchedNodes = await getAllNodes();
            setNodes(fetchedNodes);
            return fetchedNodes;
        } catch (error) {
            console.error("Failed to refetch nodes:", error);
            throw error;
        }
    }, [setNodes]);

    // Initial fetch on mount
    useEffect(() => {
        refetchNodes();
    }, [refetchNodes]);

    return {
        refetchNodes
    };
};
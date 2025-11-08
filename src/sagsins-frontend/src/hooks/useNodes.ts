import { useCallback, useEffect } from 'react';
import { useNodeStore } from '../state/nodeStore';
import { getAllNodes } from '../services/nodeService';

export const useNodes = () => {
    const { setNodes } = useNodeStore();

    // Note: WebSocket connection is now managed globally via WebSocketProvider
    // The global WebSocket will automatically update nodes via updateNodeInStore

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
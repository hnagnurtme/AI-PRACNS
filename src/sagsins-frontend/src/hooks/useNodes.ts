import { useCallback } from 'react';
import { useNodeStore } from '../state/nodeStore';
import { getAllNodes } from '../services/nodeService';

export const useNodes = () => {
    const { setNodes } = useNodeStore();

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

    return {
        refetchNodes
    };
};
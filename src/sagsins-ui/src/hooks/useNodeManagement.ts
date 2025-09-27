import { useState, useCallback } from 'react';
import type { NodeInfo } from '../types/node';
import { nodeApiService, type CreateNodeRequest, type UpdateNodeRequest } from '../services/nodeApi';

export interface UseNodeManagementReturn {
  nodes: NodeInfo[];
  loading: boolean;
  error: string | null;
  selectedNode: NodeInfo | null;
  
  // Actions
  loadNodes: () => Promise<void>;
  createNode: (nodeData: CreateNodeRequest) => Promise<NodeInfo>;
  updateNode: (nodeId: string, nodeData: Partial<UpdateNodeRequest>) => Promise<NodeInfo>;
  deleteNode: (nodeId: string) => Promise<void>;
  selectNode: (node: NodeInfo | null) => void;
  refreshNodes: () => Promise<void>;
}

export function useNodeManagement(): UseNodeManagementReturn {
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeInfo | null>(null);

  const loadNodes = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const fetchedNodes = await nodeApiService.getAllNodes();
      setNodes(fetchedNodes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load nodes');
      console.error('Error loading nodes:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const createNode = useCallback(async (nodeData: CreateNodeRequest): Promise<NodeInfo> => {
    setLoading(true);
    setError(null);
    
    try {
      const newNode = await nodeApiService.createNode(nodeData);
      setNodes(prev => [...prev, newNode]);
      return newNode;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create node';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const updateNode = useCallback(async (nodeId: string, nodeData: Partial<UpdateNodeRequest>): Promise<NodeInfo> => {
    setLoading(true);
    setError(null);
    
    try {
      const updatedNode = await nodeApiService.updateNode(nodeId, nodeData);
      setNodes(prev => prev.map(node => 
        node.nodeId === nodeId ? updatedNode : node
      ));
      
      // Update selected node if it's the one being updated
      if (selectedNode?.nodeId === nodeId) {
        setSelectedNode(updatedNode);
      }
      
      return updatedNode;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update node';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [selectedNode]);

  const deleteNode = useCallback(async (nodeId: string): Promise<void> => {
    setLoading(true);
    setError(null);
    
    try {
      await nodeApiService.deleteNode(nodeId);
      setNodes(prev => prev.filter(node => node.nodeId !== nodeId));
      
      // Clear selection if deleted node was selected
      if (selectedNode?.nodeId === nodeId) {
        setSelectedNode(null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete node';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [selectedNode]);

  const selectNode = useCallback((node: NodeInfo | null) => {
    setSelectedNode(node);
  }, []);

  const refreshNodes = useCallback(async () => {
    await loadNodes();
  }, [loadNodes]);

  return {
    nodes,
    loading,
    error,
    selectedNode,
    loadNodes,
    createNode,
    updateNode,
    deleteNode,
    selectNode,
    refreshNodes,
  };
}

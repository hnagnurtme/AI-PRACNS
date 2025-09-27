import React, { useState } from 'react';
import type { NodeInfo } from '../../types/node';
import type { CreateNodeRequest } from '../../services/nodeApi';
import { useNodeManagement } from '../../hooks/useNodeManagement';
import Topbar from './Topbar';
import Sidebar from './Sidebar';
import CesiumMap from '../cesium/CesiumMap';

interface LayoutProps {
  initialNodes: NodeInfo[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export default function Layout({ initialNodes, loading, error, onRetry }: LayoutProps) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();

  const {
    nodes,
    loading: nodeLoading,
    createNode,
    updateNode,
    deleteNode,
    selectNode
  } = useNodeManagement();

  // Initialize nodes if not loaded yet
  React.useEffect(() => {
    if (initialNodes.length > 0 && nodes.length === 0) {
      // Set initial nodes if available
      nodes.push(...initialNodes);
    }
  }, [initialNodes, nodes]);

  const handleToggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const handleNodeSelect = (nodeId: string) => {
    setSelectedNodeId(nodeId);
    const node = nodes.find(n => n.nodeId === nodeId);
    selectNode(node || null);
  };

  const handleNodeFocus = (nodeId: string) => {
    console.log('Focus on node:', nodeId);
  };

  const handleCreateNode = async (nodeData: CreateNodeRequest) => {
    try {
      await createNode(nodeData);
    } catch (error) {
      console.error('Failed to create node:', error);
    }
  };

  const handleUpdateNode = async (nodeId: string, nodeData: Partial<NodeInfo>) => {
    try {
      await updateNode(nodeId, nodeData);
    } catch (error) {
      console.error('Failed to update node:', error);
    }
  };

  const handleDeleteNode = async (nodeId: string) => {
    try {
      await deleteNode(nodeId);
      if (selectedNodeId === nodeId) {
        setSelectedNodeId(undefined);
      }
    } catch (error) {
      console.error('Failed to delete node:', error);
    }
  };

  return (
    <div className="w-screen h-screen flex flex-col bg-gray-100 overflow-hidden absolute top-0 left-0">
      {/* Topbar */}
      <Topbar 
        onToggleSidebar={handleToggleSidebar}
        isSidebarOpen={isSidebarOpen}
      />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        {isSidebarOpen && (
          <div className="flex-shrink-0">
            <Sidebar
              nodes={nodes}
              selectedNodeId={selectedNodeId}
              onNodeSelect={handleNodeSelect}
              onNodeFocus={handleNodeFocus}
              onCreateNode={handleCreateNode}
              onUpdateNode={handleUpdateNode}
              onDeleteNode={handleDeleteNode}
              loading={nodeLoading}
            />
          </div>
        )}

        {/* Map area */}
        <div className="flex-1 relative w-full h-full">
          {loading && (
            <div className="absolute inset-0 bg-black/60 z-10 flex items-center justify-center">
              <div className="text-white text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-blue-400 mx-auto mb-4"></div>
                <p>Loading satellite data...</p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 bg-black/60 z-10 flex items-center justify-center">
              <div className="bg-red-600 text-white p-6 rounded-lg max-w-md mx-4 text-center">
                <h3 className="text-lg font-semibold mb-2">❌ Error</h3>
                <p className="mb-4">{error}</p>
                <button
                  onClick={onRetry}
                  className="px-4 py-2 bg-red-700 hover:bg-red-800 rounded transition-colors"
                >
                  Retry
                </button>
              </div>
            </div>
          )}

          {!loading && !error && (
            <CesiumMap 
              nodes={nodes}
              selectedNodeId={selectedNodeId}
              onNodeFocus={handleNodeFocus}
            />
          )}
        </div>
      </div>
    </div>
  );
}

import { useState } from 'react';
import type { NodeInfo } from '../../types/node';
import type { CreateNodeRequest } from '../../services/nodeApi';
import NodeManagement from '../ui/NodeManagement';

interface SidebarProps {
  nodes: NodeInfo[];
  selectedNodeId?: string;
  onNodeSelect?: (nodeId: string) => void;
  onNodeFocus?: (nodeId: string) => void;
  onCreateNode?: (nodeData: CreateNodeRequest) => Promise<void>;
  onUpdateNode?: (nodeId: string, nodeData: Partial<NodeInfo>) => Promise<void>;
  onDeleteNode?: (nodeId: string) => Promise<void>;
  loading?: boolean;
}

export default function Sidebar({ 
  nodes, 
  selectedNodeId, 
  onNodeSelect, 
  onNodeFocus,
  onCreateNode,
  onUpdateNode,
  onDeleteNode,
  loading = false
}: SidebarProps) {
  const [activeTab, setActiveTab] = useState<'list' | 'management'>('list');
  const selectedNode = nodes.find(node => node.nodeId === selectedNodeId);

  return (
    <div className="w-80 bg-gray-900 text-white h-full flex flex-col flex-shrink-0">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Satellite Nodes</h2>
        <p className="text-sm text-gray-400 mt-1">
          {nodes.length} nodes detected
        </p>
        
        {/* Tab Navigation */}
        <div className="flex mt-3 space-x-1">
          <button
            onClick={() => setActiveTab('list')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              activeTab === 'list' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            List
          </button>
          <button
            onClick={() => setActiveTab('management')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              activeTab === 'management' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Manage
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'list' ? (
          <>
            {nodes.length === 0 ? (
              <div className="p-4 text-center text-gray-400">
                <div className="text-4xl mb-2">🛰️</div>
                <p>No nodes available</p>
              </div>
            ) : (
              <div className="p-2">
                {nodes.map((node) => (
                  <div
                    key={node.nodeId}
                    className={`
                      p-3 mb-2 rounded-lg cursor-pointer transition-all duration-200
                      ${selectedNodeId === node.nodeId 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-800 hover:bg-gray-700 text-gray-200'
                      }
                    `}
                    onClick={() => onNodeSelect?.(node.nodeId)}
                    onDoubleClick={() => onNodeFocus?.(node.nodeId)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                          node.nodeType === 'SATELLITE' ? 'bg-yellow-400' :
                          node.nodeType === 'GROUND_STATION' ? 'bg-cyan-400' :
                          node.nodeType === 'UE' ? 'bg-lime-400' :
                          node.nodeType === 'RELAY' ? 'bg-orange-400' :
                          'bg-blue-400'
                        }`}></div>
                        <div>
                          <h3 className="font-medium text-sm">{node.nodeId}</h3>
                          <p className="text-xs text-gray-400">{node.nodeType}</p>
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">
                        {node.position?.altitude ? `${Math.round(node.position.altitude)}m` : 'N/A'}
                      </div>
                    </div>
                    
                    {node.position && (
                      <div className="mt-2 text-xs text-gray-400">
                        <div>Lat: {node.position.latitude?.toFixed(4)}°</div>
                        <div>Lon: {node.position.longitude?.toFixed(4)}°</div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <NodeManagement
            nodes={nodes}
            selectedNode={selectedNode || null}
            onNodeSelect={(node) => onNodeSelect?.(node?.nodeId || '')}
            onCreateNode={onCreateNode || (() => Promise.resolve())}
            onUpdateNode={onUpdateNode || (() => Promise.resolve())}
            onDeleteNode={onDeleteNode || (() => Promise.resolve())}
            loading={loading}
          />
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <div className="text-xs text-gray-400 text-center">
          {activeTab === 'list' ? 'Double-click to focus on node' : 'Manage your satellite nodes'}
        </div>
      </div>
    </div>
  );
}

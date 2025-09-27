import { useState } from 'react';
import type { NodeInfo } from '../../types/node';
import type { CreateNodeRequest } from '../../services/nodeApi';

interface NodeManagementProps {
  nodes: NodeInfo[];
  selectedNode: NodeInfo | null;
  onNodeSelect: (node: NodeInfo | null) => void;
  onCreateNode: (nodeData: CreateNodeRequest) => Promise<void>;
  onUpdateNode: (nodeId: string, nodeData: Partial<NodeInfo>) => Promise<void>;
  onDeleteNode: (nodeId: string) => Promise<void>;
  loading: boolean;
}

export default function NodeManagement({
  nodes,
  selectedNode,
  onNodeSelect,
  onCreateNode,
  onUpdateNode,
  onDeleteNode,
  loading
}: NodeManagementProps) {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showEditForm, setShowEditForm] = useState(false);
  const [formData, setFormData] = useState<CreateNodeRequest>({
    nodeType: 'UE',
    position: {
      longitude: 0,
      latitude: 0,
      altitude: 1000
    }
  });

  const nodeTypes = ['SATELLITE', 'GROUND_STATION', 'UE', 'RELAY', 'SEA'];

  const handleCreateNode = async () => {
    try {
      await onCreateNode(formData);
      setShowCreateForm(false);
      setFormData({
        nodeType: 'UE',
        position: { longitude: 0, latitude: 0, altitude: 1000 }
      });
    } catch (error) {
      console.error('Failed to create node:', error);
    }
  };

  const handleUpdateNode = async () => {
    if (!selectedNode) return;
    
    try {
      await onUpdateNode(selectedNode.nodeId, formData);
      setShowEditForm(false);
    } catch (error) {
      console.error('Failed to update node:', error);
    }
  };

  const handleDeleteNode = async () => {
    if (!selectedNode) return;
    
    if (confirm(`Are you sure you want to delete ${selectedNode.nodeId}?`)) {
      try {
        await onDeleteNode(selectedNode.nodeId);
        onNodeSelect(null);
      } catch (error) {
        console.error('Failed to delete node:', error);
      }
    }
  };

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Node Management</h3>
        <button
          onClick={() => setShowCreateForm(true)}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
          disabled={loading}
        >
          + Add Node
        </button>
      </div>

      {/* Node List */}
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {nodes.map((node) => (
          <div
            key={node.nodeId}
            className={`p-3 rounded-lg cursor-pointer transition-colors ${
              selectedNode?.nodeId === node.nodeId
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 hover:bg-gray-700 text-gray-200'
            }`}
            onClick={() => onNodeSelect(node)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  node.nodeType === 'SATELLITE' ? 'bg-yellow-400' :
                  node.nodeType === 'GROUND_STATION' ? 'bg-cyan-400' :
                  node.nodeType === 'UE' ? 'bg-lime-400' :
                  node.nodeType === 'RELAY' ? 'bg-orange-400' :
                  'bg-blue-400'
                }`}></div>
                <div>
                  <div className="font-medium text-sm">{node.nodeId}</div>
                  <div className="text-xs opacity-75">{node.nodeType}</div>
                </div>
              </div>
              <div className="text-xs opacity-75">
                {node.position?.altitude ? `${Math.round(node.position.altitude)}m` : 'N/A'}
              </div>
            </div>
            
            {node.position && (
              <div className="mt-2 text-xs opacity-75">
                <div>Lat: {node.position.latitude?.toFixed(4)}°</div>
                <div>Lon: {node.position.longitude?.toFixed(4)}°</div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Selected Node Actions */}
      {selectedNode && (
        <div className="space-y-2">
          <div className="p-3 bg-gray-800 rounded-lg">
            <h4 className="font-medium text-white mb-2">Selected: {selectedNode.nodeId}</h4>
            <div className="text-sm text-gray-300 space-y-1">
              <div>Type: {selectedNode.nodeType}</div>
              <div>Position: {selectedNode.position?.latitude?.toFixed(4)}°, {selectedNode.position?.longitude?.toFixed(4)}°</div>
              <div>Altitude: {selectedNode.position?.altitude}m</div>
              {selectedNode.healthy !== undefined && (
                <div className={`${selectedNode.healthy ? 'text-green-400' : 'text-red-400'}`}>
                  Status: {selectedNode.healthy ? 'Healthy' : 'Unhealthy'}
                </div>
              )}
            </div>
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={() => {
                setFormData({
                  nodeType: selectedNode.nodeType,
                  position: selectedNode.position,
                  orbit: selectedNode.orbit,
                  velocity: selectedNode.velocity
                });
                setShowEditForm(true);
              }}
              className="flex-1 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded transition-colors"
              disabled={loading}
            >
              Edit
            </button>
            <button
              onClick={handleDeleteNode}
              className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
              disabled={loading}
            >
              Delete
            </button>
          </div>
        </div>
      )}

      {/* Create Node Form */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg w-96 max-w-full mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Create New Node</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Node Type</label>
                <select
                  value={formData.nodeType}
                  onChange={(e) => setFormData({...formData, nodeType: e.target.value})}
                  className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                >
                  {nodeTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Latitude</label>
                  <input
                    type="number"
                    value={formData.position.latitude}
                    onChange={(e) => setFormData({
                      ...formData,
                      position: {...formData.position, latitude: parseFloat(e.target.value) || 0}
                    })}
                    className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                    step="0.0001"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Longitude</label>
                  <input
                    type="number"
                    value={formData.position.longitude}
                    onChange={(e) => setFormData({
                      ...formData,
                      position: {...formData.position, longitude: parseFloat(e.target.value) || 0}
                    })}
                    className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                    step="0.0001"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm text-gray-300 mb-1">Altitude (m)</label>
                <input
                  type="number"
                  value={formData.position.altitude}
                  onChange={(e) => setFormData({
                    ...formData,
                    position: {...formData.position, altitude: parseFloat(e.target.value) || 1000}
                  })}
                  className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                />
              </div>
            </div>
            
            <div className="flex space-x-2 mt-6">
              <button
                onClick={handleCreateNode}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                disabled={loading}
              >
                Create
              </button>
              <button
                onClick={() => setShowCreateForm(false)}
                className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Node Form */}
      {showEditForm && selectedNode && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg w-96 max-w-full mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Edit Node: {selectedNode.nodeId}</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Node Type</label>
                <select
                  value={formData.nodeType}
                  onChange={(e) => setFormData({...formData, nodeType: e.target.value})}
                  className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                >
                  {nodeTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Latitude</label>
                  <input
                    type="number"
                    value={formData.position.latitude}
                    onChange={(e) => setFormData({
                      ...formData,
                      position: {...formData.position, latitude: parseFloat(e.target.value) || 0}
                    })}
                    className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                    step="0.0001"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Longitude</label>
                  <input
                    type="number"
                    value={formData.position.longitude}
                    onChange={(e) => setFormData({
                      ...formData,
                      position: {...formData.position, longitude: parseFloat(e.target.value) || 0}
                    })}
                    className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                    step="0.0001"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm text-gray-300 mb-1">Altitude (m)</label>
                <input
                  type="number"
                  value={formData.position.altitude}
                  onChange={(e) => setFormData({
                    ...formData,
                    position: {...formData.position, altitude: parseFloat(e.target.value) || 1000}
                  })}
                  className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                />
              </div>
            </div>
            
            <div className="flex space-x-2 mt-6">
              <button
                onClick={handleUpdateNode}
                className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded transition-colors"
                disabled={loading}
              >
                Update
              </button>
              <button
                onClick={() => setShowEditForm(false)}
                className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

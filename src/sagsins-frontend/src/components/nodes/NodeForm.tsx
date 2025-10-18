import React from 'react';

// Minimal placeholder: Node create/update is not supported in simplified API
const NodeForm: React.FC<{ onClose: () => void } & Record<string, unknown>> = ({ onClose }) => {
    return (
        <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-lg text-gray-700">
            <h2 className="text-2xl font-bold mb-6 text-gray-800">Node Management Disabled</h2>
            <p className="mb-4">Creating, updating, and deleting nodes are disabled in this build. The app now only supports:</p>
            <ul className="list-disc list-inside text-sm text-gray-700 mb-6">
                <li>Fetching all nodes</li>
                <li>Fetching node details by ID</li>
                <li>Partial update of node status via API client</li>
                <li>Health check</li>
            </ul>
            <div className="flex justify-end">
                <button 
                    type="button"
                    onClick={onClose}
                    className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-lg transition duration-150"
                >
                    Close
                </button>
            </div>
        </div>
    );
};

export default NodeForm;
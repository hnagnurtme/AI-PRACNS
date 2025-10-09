import React, { useState } from 'react';
import { useNodeStore } from '../../state/nodeStore';
import { createNode, deleteNode, updateNode } from '../../services/nodeService';
import type { CreateNodeRequest, NodeDTO, UpdateNodeRequest } from '../../types/NodeTypes';
import type { Geo3D } from '../../types/ModelTypes';

// Định nghĩa Props cho Form
interface NodeFormProps {
    onClose: () => void;
    mode: 'create' | 'update';
    initialNode?: NodeDTO; // Chỉ tồn tại khi mode là 'update'
    onSuccess?: () => void; // Callback để refresh data sau khi thành công
}

const NodeForm: React.FC<NodeFormProps> = ({ onClose, mode, initialNode, onSuccess }) => {
    const { setSelectedNode } = useNodeStore(); // Cần setSelectedNode để clear selection sau delete
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Hàm tạo ID tạm thời cho trường hợp CREATE
    const generateTempId = () => `NODE_${Math.random().toString(36).substring(2, 9).toUpperCase()}`;

    // Sử dụng state để quản lý dữ liệu form
    const [formData, setFormData] = useState<Partial<CreateNodeRequest> & Partial<UpdateNodeRequest>>(() => ({
        // THÊM: nodeId là bắt buộc khi tạo, gán giá trị tạm thời khi mode='create'
        nodeId: initialNode?.nodeId || (mode === 'create' ? generateTempId() : undefined),
        nodeType: initialNode?.nodeType || 'LEO_SATELLITE',
        isOperational: initialNode?.isOperational ?? true,
        // Chỉ lấy Geo3D, không cần Orbit/Velocity cho form đơn giản này
        position: initialNode?.position || { latitude: 0, longitude: 0, altitude: 550 },
        currentBandwidth: initialNode?.currentBandwidth || 500,
        avgLatencyMs: initialNode?.avgLatencyMs || 10,
        packetLossRate: initialNode?.packetLossRate || 0.001,
    }));

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value, type } = e.target;
        
        // Xử lý giá trị số và boolean
        let finalValue: string | number | boolean = value;

        if (type === 'number') {
            finalValue = parseFloat(value); 
        } else if (type === 'checkbox') {
            finalValue = (e.target as HTMLInputElement).checked; 
        }

        if (name === 'latitude' || name === 'longitude' || name === 'altitude') {
            // Xử lý cập nhật vị trí Geo3D
            setFormData(prev => ({
                ...prev,
                position: {
                    ...(prev.position as Geo3D),
                    [name]: finalValue as number,
                }
            }));
        } else {
            // Xử lý các trường khác
            setFormData(prev => ({ ...prev, [name]: finalValue }));
        }
    };
    
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setIsSubmitting(true);
        
        try {
            if (mode === 'create') {
                // Kiểm tra ID có bị bỏ trống không (nếu người dùng cố tình xóa)
                if (!formData.nodeId) {
                    throw new Error("Node ID is mandatory for creation.");
                }

                // Tạo Node mới
                // Ép kiểu dữ liệu vì NodeId đã được đảm bảo
                await createNode(formData as CreateNodeRequest);

            } else {
                // Cập nhật Node hiện có
                if (!initialNode) throw new Error("Initial node data is missing for update mode.");
                await updateNode(initialNode.nodeId, formData as UpdateNodeRequest);
            }
            
            onClose(); // Đóng form sau khi thành công
            onSuccess?.(); // Gọi callback để refresh data nếu có

        } catch (err) {
            // Xử lý lỗi API (đã sử dụng Type Guard)
            const errorMessage = (err as Error).message || "An unknown error occurred during API call.";
            setError("Failed to process node: " + errorMessage);
            setIsSubmitting(false);
        }
    };
    
    // Logic cho nút Xóa (chỉ hiển thị khi mode là update)
    const handleDelete = async () => {
        if (!initialNode || !window.confirm(`Are you sure you want to delete Node ${initialNode.nodeId}?`)) return;
        
        try {
            await deleteNode(initialNode.nodeId);
            setSelectedNode(null); // Clear selected node sau khi delete
            onClose(); 
            onSuccess?.(); // Gọi callback để refresh data nếu có
        } catch (err) {
            const errorMessage = (err as Error).message || "An unknown error occurred during API call.";
            setError("Failed to delete node: " + errorMessage);
        }
    }


    return (
        <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-lg text-gray-700">
            <h2 className="text-2xl font-bold mb-6 text-gray-800">
                {mode === 'create' ? 'Create New Node' : `Update Node ${initialNode?.nodeId.substring(0, 8)}...`}
            </h2>

            {error && <p className="text-red-600 mb-4 p-2 bg-red-50 rounded">{error}</p>}

            <form onSubmit={handleSubmit}>
                
                {/* 0. NODE ID (BẮT BUỘC KHI TẠO) */}
                {mode === 'create' && (
                    <FormInput 
                        label="Node ID (Mandatory)" 
                        name="nodeId" 
                        type="text" 
                        // Ép kiểu về string để phù hợp với input
                        value={formData.nodeId as string} 
                        onChange={handleChange} 
                        required 
                        disabled={false} // ID có thể chỉnh sửa khi tạo
                    />
                )}

                {/* 1. Node Type */}
                <FormGroup label="Node Type">
                    <select 
                        name="nodeType" 
                        value={formData.nodeType} 
                        onChange={handleChange}
                        className="w-full p-2 border rounded bg-gray-50"
                        required
                    >
                        <option value="LEO_SATELLITE">LEO_SATELLITE</option>
                        <option value="GROUND_STATION">GROUND_STATION</option>
                        <option value="MEO_SATELLITE">MEO_SATELLITE</option>
                        <option value="GEO_SATELLITE">GEO_SATELLITE</option>
                    </select>
                </FormGroup>

                {/* 2. Position (Geo3D) */}
                <h3 className="text-lg font-semibold mt-4 mb-2 border-b pb-1">Position (Km)</h3>
                <div className="grid grid-cols-3 gap-3">
                    <FormInput label="Latitude" name="latitude" type="number" step="any" value={formData.position?.latitude ?? 0} onChange={handleChange} required />
                    <FormInput label="Longitude" name="longitude" type="number" step="any" value={formData.position?.longitude ?? 0} onChange={handleChange} required />
                    <FormInput label="Altitude" name="altitude" type="number" step="any" value={formData.position?.altitude ?? 0} onChange={handleChange} required />
                </div>
                
                {/* 3. Metrics */}
                <h3 className="text-lg font-semibold mt-6 mb-2 border-b pb-1">Initial Metrics</h3>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                    <FormInput label="Bandwidth (Mbps)" name="currentBandwidth" type="number" step="any" value={formData.currentBandwidth ?? 0} onChange={handleChange} required />
                    <FormInput label="Latency (ms)" name="avgLatencyMs" type="number" step="any" value={formData.avgLatencyMs ?? 0} onChange={handleChange} required />
                    <FormInput label="Loss Rate (0-1)" name="packetLossRate" type="number" step="0.0001" value={formData.packetLossRate ?? 0} onChange={handleChange} required />
                </div>

                {/* 4. Operational Status */}
                <div className="mt-6">
                    <label className="flex items-center space-x-2">
                        <input 
                            type="checkbox" 
                            name="isOperational" 
                            checked={formData.isOperational ?? true}
                            onChange={handleChange}
                            className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                        />
                        <span className="text-gray-700 font-medium">Is Operational</span>
                    </label>
                </div>
                
                {/* 5. Actions */}
                <div className="mt-8 pt-4 border-t flex justify-between space-x-3">
                    {mode === 'update' && (
                        <button 
                            type="button" 
                            onClick={handleDelete}
                            className="flex-shrink-0 bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition duration-150"
                        >
                            Delete Node
                        </button>
                    )}
                    
                    <div className="flex space-x-3 ml-auto">
                        <button 
                            type="button" 
                            onClick={onClose}
                            className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-lg transition duration-150"
                            disabled={isSubmitting}
                        >
                            Cancel
                        </button>
                        <button 
                            type="submit" 
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg transition duration-150 disabled:opacity-50"
                            disabled={isSubmitting}
                        >
                            {isSubmitting ? 'Processing...' : mode === 'create' ? 'Create Node' : 'Save Changes'}
                        </button>
                    </div>
                </div>
            </form>
        </div>
    );
};

export default NodeForm;


// --- Component Phụ trợ (Tái sử dụng) ---

interface FormInputProps {
    label: string;
    name: string;
    type: string;
    value: string | number;
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
    step?: string;
    required?: boolean;
    disabled?: boolean;
}

const FormInput: React.FC<FormInputProps> = ({ label, name, type, value, onChange, step, required, disabled }) => (
    <FormGroup label={label}>
        <input
            type={type}
            name={name}
            value={value}
            onChange={onChange}
            step={step}
            required={required}
            disabled={disabled}
            className="w-full p-2 border rounded bg-gray-50 **text-gray-900** focus:border-indigo-500 disabled:bg-gray-100"
        />
    </FormGroup>
);

const FormGroup: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
    <div className="mb-4">
        <label className="block text-gray-700 text-sm font-bold mb-1">
            {label}
        </label>
        {children}
    </div>
);
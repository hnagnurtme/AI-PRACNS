import axios from 'axios';
import type { CreateNodeRequest, NodeDTO, UpdateNodeRequest } from '../types/NodeTypes';


const API_BASE_URL = 'http://localhost:8080/api/nodes'; 


// --- READ ---
export const getAllNodes = async (): Promise<NodeDTO[]> => {
    // axios.get<NodeDTO[]> đảm bảo kiểu trả về là NodeDTO[]
    const response = await axios.get<NodeDTO[]>(API_BASE_URL);
    return response.data;
};

export const getNodeById = async (nodeId: string): Promise<NodeDTO> => {
    const response = await axios.get<NodeDTO>(`${API_BASE_URL}/${nodeId}`);
    return response.data;
};


// --- CREATE ---
export const createNode = async (nodeData: CreateNodeRequest): Promise<NodeDTO> => {
    const response = await axios.post<NodeDTO>(API_BASE_URL, nodeData);
    return response.data;
};

// --- UPDATE ---
export const updateNode = async (nodeId: string, updateData: UpdateNodeRequest): Promise<NodeDTO> => {
    // Sử dụng PUT/PATCH tùy theo API backend của bạn
    const response = await axios.put<NodeDTO>(`${API_BASE_URL}/${nodeId}`, updateData);
    return response.data;
};

// --- DELETE ---
export const deleteNode = async (nodeId: string): Promise<void> => {
    await axios.delete(`${API_BASE_URL}/${nodeId}`);
};
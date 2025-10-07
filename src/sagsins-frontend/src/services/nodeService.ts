// src/services/nodeService.ts

import axios from 'axios';
import type { CreateNodeRequest, NodeDTO, UpdateNodeRequest, ApiResponse } from '../types/NodeTypes';


const API_BASE_URL = 'http://localhost:8080/api/v1/nodes'; 

// Hàm tiện ích để trích xuất dữ liệu và xử lý lỗi
const extractData = <T>(response: ApiResponse<T>): T => {
    if (response.status !== 200 || response.error) {
        // Ném lỗi nếu status không phải 200 hoặc có trường 'error'
        throw new Error(response.message || `API call failed with status ${response.status}`);
    }
    return response.data;
};


// --- READ ALL ---
export const getAllNodes = async (): Promise<NodeDTO[]> => {
    // 1. Khai báo response sẽ là kiểu ApiResponse<NodeDTO[]>
    const response = await axios.get<ApiResponse<NodeDTO[]>>(API_BASE_URL);
    
    // 2. Trích xuất mảng NodeDTO[] từ trường 'data'
    return extractData(response.data);
};

// --- READ BY ID ---
export const getNodeById = async (nodeId: string): Promise<NodeDTO> => {
    // 1. Khai báo response sẽ là kiểu ApiResponse<NodeDTO>
    const response = await axios.get<ApiResponse<NodeDTO>>(`${API_BASE_URL}/${nodeId}`);
    
    // 2. Trích xuất NodeDTO từ trường 'data'
    return extractData(response.data);
};


// --- CREATE ---
export const createNode = async (nodeData: CreateNodeRequest): Promise<NodeDTO> => {
    const response = await axios.post<ApiResponse<NodeDTO>>(API_BASE_URL, nodeData);
    return extractData(response.data);
};

// --- UPDATE ---
export const updateNode = async (nodeId: string, updateData: UpdateNodeRequest): Promise<NodeDTO> => {
    const response = await axios.put<ApiResponse<NodeDTO>>(`${API_BASE_URL}/${nodeId}`, updateData);
    return extractData(response.data);
};

// --- DELETE ---
export const deleteNode = async (nodeId: string): Promise<void> => {
    // Đối với DELETE, API thường trả về mã 204 No Content, 
    // nhưng nếu backend vẫn trả về cấu trúc ApiResponse, ta vẫn xử lý để kiểm tra lỗi.
    const response = await axios.delete<ApiResponse<null>>(`${API_BASE_URL}/${nodeId}`);
    
    // Kiểm tra lỗi, nhưng không cần trả về dữ liệu (void)
    extractData(response.data); 
};
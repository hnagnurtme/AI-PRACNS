// src/services/nodeService.ts

import axios, { AxiosError } from 'axios';
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

// Hàm xử lý lỗi axios
const handleAxiosError = (error: unknown): Error => {
    if (error instanceof AxiosError) {
        if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
            return new Error('Cannot connect to server. Please make sure the backend is running on port 8080.');
        }
        
        if (error.response) {
            // Server trả về response với status code khác 2xx
            return new Error(`Server error: ${error.response.status} - ${error.response.data?.message || error.message}`);
        } else if (error.request) {
            // Request được gửi nhưng không nhận được response
            return new Error('No response from server. Please check your connection.');
        }
    }
    
    if (error instanceof Error) {
        return error;
    }
    
    return new Error('An unexpected error occurred.');
};


// --- READ ALL ---
export const getAllNodes = async (): Promise<NodeDTO[]> => {
    try {
        // 1. Khai báo response sẽ là kiểu ApiResponse<NodeDTO[]>
        const response = await axios.get<ApiResponse<NodeDTO[]>>(API_BASE_URL);
        
        // 2. Trích xuất mảng NodeDTO[] từ trường 'data'
        return extractData(response.data);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- READ BY ID ---
export const getNodeById = async (nodeId: string): Promise<NodeDTO> => {
    try {
        // 1. Khai báo response sẽ là kiểu ApiResponse<NodeDTO>
        const response = await axios.get<ApiResponse<NodeDTO>>(`${API_BASE_URL}/${nodeId}`);
        
        // 2. Trích xuất NodeDTO từ trường 'data'
        return extractData(response.data);
    } catch (error) {
        throw handleAxiosError(error);
    }
};


// --- CREATE ---
export const createNode = async (nodeData: CreateNodeRequest): Promise<NodeDTO> => {
    try {
        const response = await axios.post<ApiResponse<NodeDTO>>(API_BASE_URL, nodeData);
        return extractData(response.data);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- UPDATE ---
export const updateNode = async (nodeId: string, updateData: UpdateNodeRequest): Promise<NodeDTO> => {
    try {
        const response = await axios.put<ApiResponse<NodeDTO>>(`${API_BASE_URL}/${nodeId}`, updateData);
        return extractData(response.data);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- DELETE ---
export const deleteNode = async (nodeId: string): Promise<void> => {
    try {
        // Đối với DELETE, API thường trả về mã 204 No Content, 
        // nhưng nếu backend vẫn trả về cấu trúc ApiResponse, ta vẫn xử lý để kiểm tra lỗi.
        const response = await axios.delete<ApiResponse<null>>(`${API_BASE_URL}/${nodeId}`);
        
        // Kiểm tra lỗi, nhưng không cần trả về dữ liệu (void)
        extractData(response.data);
    } catch (error) {
        throw handleAxiosError(error);
    }
};
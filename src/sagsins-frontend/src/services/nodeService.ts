// src/services/nodeService.ts

import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type { 
    CreateNodeRequest, 
    NodeDTO, 
    UpdateNodeRequest, 
    HealthResponse, 
    DockerResponse 
} from '../types/NodeTypes';

const NODES_ENDPOINT = '/api/v1/nodes'; 
const HEALTH_ENDPOINT = '/health';
const DOCKER_ENDPOINT = '/api/v1/docker/allLinks';

// API Response wrapper (for backward compatibility)
interface ApiResponse<T> {
    status: number;
    error: string | null;
    message: string;
    data: T;
}

// Hàm tiện ích để trích xuất dữ liệu và xử lý lỗi
const extractData = <T>(response: ApiResponse<T>): T => {
    if (response.status !== 200 || response.error) {
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
        const response = await axiosClient.get<NodeDTO[] | ApiResponse<NodeDTO[]>>(NODES_ENDPOINT);
        console.log('API Response:', response.data);
        
        // Check if response is wrapped in ApiResponse format
        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<NodeDTO[]>);
        }
        
        // Direct array response
        return response.data as NodeDTO[];
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- READ BY ID ---
export const getNodeById = async (nodeId: string): Promise<NodeDTO> => {
    try {
        const response = await axiosClient.get<NodeDTO | ApiResponse<NodeDTO>>(`${NODES_ENDPOINT}/${nodeId}`);
        
        // Check if response is wrapped in ApiResponse format
        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<NodeDTO>);
        }
        
        return response.data as NodeDTO;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- CREATE ---
export const createNode = async (nodeData: CreateNodeRequest): Promise<NodeDTO> => {
    try {
        const response = await axiosClient.post<NodeDTO | ApiResponse<NodeDTO>>(NODES_ENDPOINT, nodeData);
        
        // Check if response is wrapped in ApiResponse format
        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<NodeDTO>);
        }
        
        return response.data as NodeDTO;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- UPDATE ---
export const updateNode = async (nodeId: string, updateData: UpdateNodeRequest): Promise<NodeDTO> => {
    try {
        console.log('Updating node with data:', updateData);
        const response = await axiosClient.patch<NodeDTO | ApiResponse<NodeDTO>>(`${NODES_ENDPOINT}/${nodeId}`, updateData);
        console.log('API Response:', response.data);
        
        // Check if response is wrapped in ApiResponse format
        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<NodeDTO>);
        }
        
        return response.data as NodeDTO;
    } catch (error) {
        console.log('Error during API call:', error);
        throw handleAxiosError(error);
    }
};

// --- DELETE ---
export const deleteNode = async (nodeId: string): Promise<void> => {
    try {
        await axiosClient.delete(`${NODES_ENDPOINT}/${nodeId}`);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- RUN NODE PROCESS ---
export const runNodeProcess = async (nodeId: string): Promise<void> => {
    try {
        await axiosClient.post(`${NODES_ENDPOINT}/run/${nodeId}`);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- HEALTH CHECK ---
export const checkHealth = async (): Promise<HealthResponse> => {
    try {
        const response = await axiosClient.get<HealthResponse>(HEALTH_ENDPOINT);
        return response.data;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

// --- GET DOCKER ENTITIES ---
export const getDockerEntities = async (isRunning: boolean): Promise<DockerResponse[]> => {
    try {
        const response = await axiosClient.get<DockerResponse[]>(DOCKER_ENDPOINT, {
            params: { isRunning }
        });
        return response.data;
    } catch (error) {
        throw handleAxiosError(error);
    }
};
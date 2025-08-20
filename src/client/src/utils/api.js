import axios from 'axios';
import { API_ENDPOINTS } from './config';

// Create axios instance with default configuration
const apiClient = axios.create({
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API helper functions
export const api = {
  // System status
  getSystemStatus: () => apiClient.get(API_ENDPOINTS.STATUS),
  
  // Node management
  getNodes: () => apiClient.get(API_ENDPOINTS.NODES),
  getNode: (nodeId) => apiClient.get(`${API_ENDPOINTS.NODES}/${nodeId}`),
  
  // Request management
  createRequest: (requestData) => apiClient.post(API_ENDPOINTS.REQUESTS, requestData),
  getRequests: () => apiClient.get(API_ENDPOINTS.REQUESTS),
  getRequest: (requestId) => apiClient.get(`${API_ENDPOINTS.REQUESTS}/${requestId}`),
  
  // AI optimization
  optimizeAllocation: (data) => apiClient.post(API_ENDPOINTS.AI_OPTIMIZE, data),
  predictPerformance: (data) => apiClient.post(API_ENDPOINTS.AI_PREDICT, data),
  getStrategies: () => apiClient.get(API_ENDPOINTS.AI_STRATEGIES),
};

export default apiClient;
// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080/api';
const AI_SERVER_URL = process.env.REACT_APP_AI_SERVER_URL || 'http://localhost:8081/api';

export const API_ENDPOINTS = {
  // Resource Allocation Server endpoints
  NODES: `${API_BASE_URL}/nodes`,
  REQUESTS: `${API_BASE_URL}/requests`,
  STATUS: `${API_BASE_URL}/status`,
  
  // AI Server endpoints
  AI_OPTIMIZE: `${AI_SERVER_URL}/optimize`,
  AI_PREDICT: `${AI_SERVER_URL}/predict`,
  AI_STRATEGIES: `${AI_SERVER_URL}/strategies`,
};

export const APP_CONFIG = {
  NAME: process.env.REACT_APP_NAME || 'AI-PRACNS Client',
  VERSION: process.env.REACT_APP_VERSION || '1.0.0',
  DEBUG: process.env.REACT_APP_DEBUG === 'true',
};
import type { NodeInfo } from '../types/node';

const API_BASE_URL = 'http://localhost:8080/api';

export interface CreateNodeRequest {
  nodeType: string;
  position: {
    longitude: number;
    latitude: number;
    altitude: number;
  };
  orbit?: {
    altitude: number;
    inclination: number;
  };
  velocity?: {
    speed: number;
  };
}

export interface UpdateNodeRequest {
  nodeId: string;
  nodeType?: string;
  position?: {
    longitude: number;
    latitude: number;
    altitude: number;
  };
  orbit?: {
    altitude: number;
    inclination: number;
  };
  velocity?: {
    speed: number;
  };
}

class NodeApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Get all nodes
  async getAllNodes(): Promise<NodeInfo[]> {
    return this.request<NodeInfo[]>('/nodes');
  }

  // Create new node
  async createNode(nodeData: CreateNodeRequest): Promise<NodeInfo> {
    return this.request<NodeInfo>('/nodes', {
      method: 'POST',
      body: JSON.stringify(nodeData),
    });
  }

  // Update existing node
  async updateNode(nodeId: string, nodeData: Partial<UpdateNodeRequest>): Promise<NodeInfo> {
    return this.request<NodeInfo>(`/nodes/${nodeId}`, {
      method: 'PUT',
      body: JSON.stringify({
        nodeId,
        ...nodeData
      }),
    });
  }

  // Delete node
  async deleteNode(nodeId: string): Promise<void> {
    await this.request<void>(`/nodes/${nodeId}`, {
      method: 'DELETE',
    });
  }

  // Get node by ID
  async getNodeById(nodeId: string): Promise<NodeInfo> {
    return this.request<NodeInfo>(`/nodes/${nodeId}`);
  }
}

export const nodeApiService = new NodeApiService();

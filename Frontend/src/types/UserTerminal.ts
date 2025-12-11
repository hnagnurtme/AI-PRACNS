import type { Position, QoS } from './ComparisonTypes';

export type TerminalStatus = 'idle' | 'connecting' | 'connected' | 'transmitting' | 'disconnected' | 'error';

export interface UserTerminal {
  terminalId: string;
  name: string;
  position: Position;
  qosRequirements: QoS;
  connectedNodeId?: string | null;
  status: TerminalStatus;
  metadata: {
    type: 'mobile' | 'fixed' | 'portable';
    region?: string;
    priority?: number;
  };
  connectionMetrics?: {
    latency?: number;
    bandwidth?: number;
    signalStrength?: number;
    connectedAt?: string;
  };
  createdAt: string;
  lastUpdated: string;
}

export interface TerminalConnectionResult {
  terminalId: string;
  success: boolean;
  connectedNodeId?: string | null;
  latency?: number;
  bandwidth?: number;
  error?: string;
  timestamp: string;
}

export interface GenerateTerminalsRequest {
  count: number;
  bounds?: {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  };
  region?: string;
  distribution?: 'random' | 'uniform' | 'clustered';
}

export interface GenerateTerminalsResponse {
  terminals: UserTerminal[];
  totalGenerated: number;
  timestamp: string;
}

export interface ConnectTerminalRequest {
  terminalId: string;
  nodeId?: string; // Optional: if not provided, backend will find best node
  algorithm?: 'dijkstra' | 'rl' | 'auto';
}

export interface ConnectTerminalResponse {
  success: boolean;
  terminalId: string;
  connectedNodeId?: string | null;
  connectionResult?: TerminalConnectionResult;
  error?: string;
}


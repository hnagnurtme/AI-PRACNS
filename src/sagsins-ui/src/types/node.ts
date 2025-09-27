export interface NodeInfo {
  nodeId: string;
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
  linkAvailable?: boolean;
  bandwidth?: number;
  latencyMs?: number;
  packetLossRate?: number;
  bufferSize?: number;
  throughput?: number;
  lastUpdated?: number;
  healthy?: boolean;
}

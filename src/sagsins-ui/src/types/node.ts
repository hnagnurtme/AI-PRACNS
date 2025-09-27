export interface NodeInfo {
  nodeId: string;
  nodeType: string;
  position: {
    longitude: number;
    latitude: number;
    altitude?: number;
  };
  orbit?: {
    inclination?: number;
    altitude?: number;
  };
  velocity?: {
    speed?: number;
  };
  healthy?: boolean;
}

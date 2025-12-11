import { useState, useEffect, useCallback } from 'react';
import { getTopology, getTopologyStatistics, getTopologyConnections } from '../services/topologyService';
import type { NetworkTopology, NetworkStatistics, NetworkConnection, TopologyUpdate } from '../types/NetworkTopologyTypes';
import { useWebSocket } from '../contexts/WebSocketContext';

interface UseNetworkTopologyReturn {
    topology: NetworkTopology | null;
    statistics: NetworkStatistics | null;
    connections: NetworkConnection[];
    loading: boolean;
    error: Error | null;
    refetch: () => Promise<void>;
    refetchStatistics: () => Promise<void>;
    refetchConnections: () => Promise<void>;
}

export const useNetworkTopology = (): UseNetworkTopologyReturn => {
    const [topology, setTopology] = useState<NetworkTopology | null>(null);
    const [statistics, setStatistics] = useState<NetworkStatistics | null>(null);
    const [connections, setConnections] = useState<NetworkConnection[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<Error | null>(null);
    const { isConnected, subscribeToTopologyUpdates } = useWebSocket();

    const refetch = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await getTopology();
            setTopology(data);
            setStatistics(data.statistics);
            setConnections(data.connections);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to fetch topology');
            setError(error);
            console.error('Failed to fetch topology:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    const refetchStatistics = useCallback(async () => {
        try {
            const data = await getTopologyStatistics();
            setStatistics(data);
            if (topology) {
                setTopology({ ...topology, statistics: data });
            }
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to fetch statistics');
            setError(error);
            console.error('Failed to fetch statistics:', error);
        }
    }, [topology]);

    const refetchConnections = useCallback(async () => {
        try {
            const data = await getTopologyConnections();
            setConnections(data);
            if (topology) {
                setTopology({ ...topology, connections: data });
            }
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to fetch connections');
            setError(error);
            console.error('Failed to fetch connections:', error);
        }
    }, [topology]);

    // Initial fetch
    useEffect(() => {
        refetch();
    }, [refetch]);

    // Subscribe to WebSocket updates
    useEffect(() => {
        if (!isConnected) return;

        const unsubscribe = subscribeToTopologyUpdates((update: TopologyUpdate) => {
            console.log('📡 Topology update received:', update);
            
            if (update.type === 'statistics' && update.data) {
                setStatistics(update.data as NetworkStatistics);
                if (topology) {
                    setTopology({ ...topology, statistics: update.data as NetworkStatistics });
                }
            } else if (update.type === 'connection' && update.data) {
                const connection = update.data as NetworkConnection;
                setConnections((prev) => {
                    const index = prev.findIndex(
                        c => c.fromNodeId === connection.fromNodeId && c.toNodeId === connection.toNodeId
                    );
                    if (index >= 0) {
                        const updated = [...prev];
                        updated[index] = connection;
                        return updated;
                    }
                    return [...prev, connection];
                });
            } else if (update.type === 'node' && update.data) {
                // Update node in topology
                if (topology) {
                    const updatedNodes = topology.nodes.map(node => 
                        node.nodeId === (update.data as any).nodeId ? { ...node, ...update.data } : node
                    );
                    setTopology({ ...topology, nodes: updatedNodes });
                }
            }
        });

        return unsubscribe;
    }, [isConnected, subscribeToTopologyUpdates, topology]);

    return {
        topology,
        statistics,
        connections,
        loading,
        error,
        refetch,
        refetchStatistics,
        refetchConnections,
    };
};


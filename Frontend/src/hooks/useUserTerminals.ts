import { useState, useCallback, useEffect } from 'react';
import {
    getUserTerminals,
    generateUserTerminals,
    connectTerminalToNode,
    disconnectTerminal,
    deleteTerminal,
    deleteAllTerminals,
} from '../services/userTerminalService';
import { useTerminalStore } from '../state/terminalStore';
import type {
    UserTerminal,
    GenerateTerminalsRequest,
    ConnectTerminalRequest,
    TerminalConnectionResult,
} from '../types/UserTerminalTypes';

interface UseUserTerminalsReturn {
    terminals: UserTerminal[];
    loading: boolean;
    error: Error | null;
    refetchTerminals: () => Promise<void>;
    generateTerminals: (request: GenerateTerminalsRequest) => Promise<UserTerminal[]>;
    connectTerminal: (request: ConnectTerminalRequest) => Promise<TerminalConnectionResult>;
    disconnectTerminalById: (terminalId: string) => Promise<void>;
    removeTerminal: (terminalId: string) => Promise<void>;
    clearAllTerminals: () => Promise<void>;
}

export const useUserTerminals = (): UseUserTerminalsReturn => {
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<Error | null>(null);
    const { terminals, setTerminals, updateTerminalInStore } = useTerminalStore();

    const refetchTerminals = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const fetchedTerminals = await getUserTerminals();
            setTerminals(fetchedTerminals);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to fetch terminals');
            setError(error);
            console.error('Failed to refetch terminals:', error);
        } finally {
            setLoading(false);
        }
    }, [setTerminals]);

    const generateTerminals = useCallback(
        async (request: GenerateTerminalsRequest): Promise<UserTerminal[]> => {
            setLoading(true);
            setError(null);
            try {
                const newTerminals = await generateUserTerminals(request);
                // Refresh the list after generation
                await refetchTerminals();
                return newTerminals;
            } catch (err) {
                const error = err instanceof Error ? err : new Error('Failed to generate terminals');
                setError(error);
                console.error('Failed to generate terminals:', error);
                throw error;
            } finally {
                setLoading(false);
            }
        },
        [refetchTerminals]
    );

    const connectTerminal = useCallback(
        async (request: ConnectTerminalRequest): Promise<TerminalConnectionResult> => {
            setError(null);
            try {
                const result = await connectTerminalToNode(request);
                // Update terminal in store
                const terminal = terminals.find(t => t.terminalId === request.terminalId);
                if (terminal) {
                    updateTerminalInStore({
                        ...terminal,
                        status: result.success ? 'connected' : 'idle',
                        connectedNodeId: result.success ? request.nodeId : null,
                        connectionMetrics: result.success
                            ? {
                                  latencyMs: result.latencyMs,
                                  bandwidthMbps: result.bandwidthMbps,
                                  packetLossRate: result.packetLossRate,
                              }
                            : undefined,
                        lastUpdated: result.timestamp,
                    });
                }
                return result;
            } catch (err) {
                const error = err instanceof Error ? err : new Error('Failed to connect terminal');
                setError(error);
                console.error('Failed to connect terminal:', error);
                throw error;
            }
        },
        [terminals, updateTerminalInStore]
    );

    const disconnectTerminalById = useCallback(
        async (terminalId: string): Promise<void> => {
            setError(null);
            try {
                await disconnectTerminal(terminalId);
                // Update terminal in store
                const terminal = terminals.find(t => t.terminalId === terminalId);
                if (terminal) {
                    updateTerminalInStore({
                        ...terminal,
                        status: 'idle',
                        connectedNodeId: null,
                        connectionMetrics: undefined,
                        lastUpdated: new Date().toISOString(),
                    });
                }
            } catch (err) {
                const error = err instanceof Error ? err : new Error('Failed to disconnect terminal');
                setError(error);
                console.error('Failed to disconnect terminal:', error);
                throw error;
            }
        },
        [terminals, updateTerminalInStore]
    );

    const removeTerminal = useCallback(
        async (terminalId: string): Promise<void> => {
            setError(null);
            try {
                await deleteTerminal(terminalId);
                // Remove terminal from store
                setTerminals(terminals.filter((term) => term.terminalId !== terminalId));
            } catch (err) {
                const error = err instanceof Error ? err : new Error('Failed to delete terminal');
                setError(error);
                console.error('Failed to delete terminal:', error);
                throw error;
            }
        },
        [terminals, setTerminals]
    );

    const clearAllTerminals = useCallback(async (): Promise<void> => {
        setLoading(true);
        setError(null);
        try {
            await deleteAllTerminals();
            setTerminals([]);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to clear terminals');
            setError(error);
            console.error('Failed to clear terminals:', error);
            throw error;
        } finally {
            setLoading(false);
        }
    }, [setTerminals]);

    // Initial fetch on mount
    useEffect(() => {
        refetchTerminals();
    }, [refetchTerminals]);

    return {
        terminals,
        loading,
        error,
        refetchTerminals,
        generateTerminals,
        connectTerminal,
        disconnectTerminalById,
        removeTerminal,
        clearAllTerminals,
    };
};


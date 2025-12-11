import React, { createContext, useContext, useEffect, useRef, useCallback } from 'react';
import type { ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import type { NodeDTO } from '../types/NodeTypes';
import type { TerminalUpdate, TerminalConnectionResult } from '../types/UserTerminalTypes';
import type { TopologyUpdate } from '../types/NetworkTopologyTypes';
import { useNodeStore } from '../state/nodeStore';

interface WebSocketContextType {
    isConnected: boolean;
    socket: React.MutableRefObject<Socket | null>;
    subscribeToTerminalUpdates: (callback: (update: TerminalUpdate) => void) => () => void;
    subscribeToTerminalConnections: (callback: (result: TerminalConnectionResult) => void) => () => void;
    subscribeToTopologyUpdates: (callback: (update: TopologyUpdate) => void) => () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
    children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
    const socketRef = useRef<Socket | null>(null);
    const [isConnected, setIsConnected] = React.useState(false);
    const { updateNodeInStore } = useNodeStore();
    
    // Store callbacks for terminal updates
    const terminalUpdateCallbacksRef = useRef<Set<(update: TerminalUpdate) => void>>(new Set());
    const terminalConnectionCallbacksRef = useRef<Set<(result: TerminalConnectionResult) => void>>(new Set());
    const topologyUpdateCallbacksRef = useRef<Set<(update: TopologyUpdate) => void>>(new Set());

    // Subscribe to terminal updates
    const subscribeToTerminalUpdates = useCallback((callback: (update: TerminalUpdate) => void) => {
        terminalUpdateCallbacksRef.current.add(callback);
        
        return () => {
            terminalUpdateCallbacksRef.current.delete(callback);
        };
    }, []);

    // Subscribe to terminal connection results
    const subscribeToTerminalConnections = useCallback((callback: (result: TerminalConnectionResult) => void) => {
        terminalConnectionCallbacksRef.current.add(callback);
        
        return () => {
            terminalConnectionCallbacksRef.current.delete(callback);
        };
    }, []);

    // Subscribe to topology updates
    const subscribeToTopologyUpdates = useCallback((callback: (update: TopologyUpdate) => void) => {
        topologyUpdateCallbacksRef.current.add(callback);
        
        return () => {
            topologyUpdateCallbacksRef.current.delete(callback);
        };
    }, []);

    useEffect(() => {
        const wsUrl = import.meta.env.VITE_WS_URL || 'http://localhost:8080';
        console.log('🔌 Initializing Socket.IO connection to:', wsUrl);

        const socket = io(wsUrl, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 5000,
            reconnectionAttempts: 10,
        });

        socket.on('connect', () => {
            console.log('✅ Socket.IO connected');
            setIsConnected(true);
        });

        socket.on('disconnect', () => {
            console.log('🔌 Socket.IO disconnected');
            setIsConnected(false);
        });

        socket.on('connect_error', (error) => {
            console.warn('⚠️ Socket.IO connection error:', error);
        });

        // Subscribe to node status updates
        socket.on('node-status', (nodeUpdate: NodeDTO) => {
            console.log('📩 Node status update received:', nodeUpdate);
            updateNodeInStore(nodeUpdate);
        });

        // Subscribe to terminal updates
        socket.on('terminal-updates', (terminalUpdate: TerminalUpdate) => {
            console.log('📩 Terminal update received:', terminalUpdate);
            terminalUpdateCallbacksRef.current.forEach((callback) => {
                callback(terminalUpdate);
            });
        });

        // Subscribe to terminal connection results
        socket.on('terminal-connections', (connectionResult: TerminalConnectionResult) => {
            console.log('📩 Terminal connection result received:', connectionResult);
            terminalConnectionCallbacksRef.current.forEach((callback) => {
                callback(connectionResult);
            });
        });

        // Subscribe to topology updates
        socket.on('network-topology', (topologyUpdate: TopologyUpdate) => {
            console.log('📩 Topology update received:', topologyUpdate);
            topologyUpdateCallbacksRef.current.forEach((callback) => {
                callback(topologyUpdate);
            });
        });

        socketRef.current = socket;

        // Cleanup on unmount
        return () => {
            if (socketRef.current) {
                console.log('🔌 Disconnecting Socket.IO');
                socketRef.current.disconnect();
            }
        };
    }, [updateNodeInStore]);

    return (
        <WebSocketContext.Provider
            value={{
                isConnected,
                socket: socketRef,
                subscribeToTerminalUpdates,
                subscribeToTerminalConnections,
                subscribeToTopologyUpdates,
            }}
        >
            {children}
        </WebSocketContext.Provider>
    );
};

export const useWebSocket = () => {
    const context = useContext(WebSocketContext);
    if (context === undefined) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    return context;
};

export const useWebSocketContext = () => {
    return useWebSocket();
};

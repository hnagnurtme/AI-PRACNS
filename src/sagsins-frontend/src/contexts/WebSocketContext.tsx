import React, { createContext, useContext, useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Client } from '@stomp/stompjs';
import SockJS from 'sockjs-client';
import type { NodeDTO } from '../types/NodeTypes';
import { useNodeStore } from '../state/nodeStore';

interface WebSocketContextType {
    isConnected: boolean;
    client: React.MutableRefObject<Client | null>;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
    children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
    const clientRef = useRef<Client | null>(null);
    const [isConnected, setIsConnected] = React.useState(false);
    const { updateNodeInStore } = useNodeStore();

    useEffect(() => {
        const wsUrl = import.meta.env.VITE_WS_URL || 'http://localhost:8080/ws';
        console.log('🔌 Initializing global WebSocket connection to:', wsUrl);

        const socket = new SockJS(wsUrl);
        const client = new Client({
            webSocketFactory: () => socket as unknown as WebSocket,
            onConnect: () => {
                console.log('✅ Global WebSocket connected');
                setIsConnected(true);

                // Subscribe to node status updates
                client.subscribe('/topic/node-status', (msg) => {
                    console.log('📩 Node status update received (global):', msg);
                    const nodeUpdate: NodeDTO = JSON.parse(msg.body);
                    updateNodeInStore(nodeUpdate);
                });

                // Subscribe to other topics as needed
                // client.subscribe('/topic/packets', (msg) => { ... });
            },
            onDisconnect: () => {
                console.log('🔌 Global WebSocket disconnected');
                setIsConnected(false);
            },
            onStompError: (err) => {
                console.error('❌ Global WebSocket STOMP error:', err);
                setIsConnected(false);
            },
            // Reconnection configuration
            reconnectDelay: 5000,
            heartbeatIncoming: 4000,
            heartbeatOutgoing: 4000,
        });

        client.activate();
        clientRef.current = client;

        // Cleanup on unmount
        return () => {
            if (clientRef.current) {
                console.log('🔌 Deactivating global WebSocket connection');
                clientRef.current.deactivate();
            }
        };
    }, []); // Empty dependency array means this runs once on mount

    return (
        <WebSocketContext.Provider value={{ isConnected, client: clientRef }}>
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

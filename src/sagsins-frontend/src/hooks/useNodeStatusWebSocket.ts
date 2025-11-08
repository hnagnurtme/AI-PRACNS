// src/hooks/useNodeStatusWebSocket.ts
import { useEffect, useRef } from "react";
import { Client } from "@stomp/stompjs";
import SockJS from "sockjs-client";
import type { NodeDTO } from "../types/NodeTypes";

interface UseNodeStatusWebSocketProps {
    url: string;
    onNodeUpdate: (node: NodeDTO) => void;
}

export const useNodeStatusWebSocket = ({ url, onNodeUpdate }: UseNodeStatusWebSocketProps) => {
    const clientRef = useRef<Client | null>(null);

    useEffect(() => {
        const socket = new SockJS(url);
        const client = new Client({
            webSocketFactory: () => socket as unknown as WebSocket,
            onConnect: () => {
                console.log("✅ Connected to Node Status WebSocket");
                client.subscribe("/topic/node-status", (msg) => {
                    console.log("📩 Node status update received", msg);
                    const nodeUpdate: NodeDTO = JSON.parse(msg.body);
                    onNodeUpdate(nodeUpdate);
                });
            },
            onStompError: (err) => console.error("❌ Node Status STOMP error:", err),
            onDisconnect: () => {
                console.log("🔌 Node Status WebSocket disconnected");
            },
            // Reconnection configuration
            reconnectDelay: 5000,
            heartbeatIncoming: 4000,
            heartbeatOutgoing: 4000,
        });

        client.activate();
        clientRef.current = client;

        return () => {
            if (clientRef.current) {
                console.log("🔌 Deactivating Node Status WebSocket");
                clientRef.current.deactivate();
            }
        };
    }, [url, onNodeUpdate]);

    return clientRef;
};

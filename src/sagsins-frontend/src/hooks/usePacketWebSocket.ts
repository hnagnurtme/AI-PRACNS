// src/hooks/usePacketWebSocket.ts
import { useEffect, useState } from "react";
import { Client } from "@stomp/stompjs";
import SockJS from "sockjs-client";
import type { ComparisonData } from "../types/ComparisonTypes";

export const usePacketWebSocket = ( url: string ) => {
    const [ messages, setMessages ] = useState<ComparisonData[]>( [] );

    useEffect( () => {
        const socket = new SockJS( url );
        const client = new Client( {
            webSocketFactory: () => socket as unknown as WebSocket,
            onConnect: () => {
                console.log( "✅ Connected to Packet WebSocket" );
                client.subscribe( "/topic/packets", ( msg ) => {
                    console.log( "📩 Packet message received", msg );
                    try {
                        const body: ComparisonData = JSON.parse( msg.body );
                        setMessages( ( prev ) => [ ...prev, body ] );
                    } catch (error) {
                        console.error("❌ Error parsing packet message:", error);
                    }
                } );
            },
            onStompError: ( err ) => {
                console.error( "❌ Packet WebSocket STOMP error:", err );
            },
            onDisconnect: () => {
                console.log("🔌 Packet WebSocket disconnected");
            },
            // Add reconnection configuration
            reconnectDelay: 5000,
            heartbeatIncoming: 4000,
            heartbeatOutgoing: 4000,
        } );

        client.activate();
        return () => {
            console.log("🔌 Deactivating Packet WebSocket");
            client.deactivate();
        };
    }, [ url ] );

    return messages;
};

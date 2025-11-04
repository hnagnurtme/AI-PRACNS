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
                console.log( "✅ Connected to WebSocket" );
                client.subscribe( "/topic/packets", ( msg ) => {
                    console.log( "📩 Message received", msg );
                    const body: ComparisonData = JSON.parse( msg.body );

                    
                    setMessages( ( prev ) => [ ...prev, body ] );
                } );
            },
            onStompError: ( err ) => console.error( "STOMP error:", err ),
        } );

        client.activate();
        return () => {
            client.deactivate();
        };
    }, [ url ] );

    return messages;
};

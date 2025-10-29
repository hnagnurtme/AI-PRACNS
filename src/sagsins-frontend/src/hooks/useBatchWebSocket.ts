// src/hooks/useBatchWebSocket.ts (Phiên bản ĐÃ SỬA LỖI)

import { useEffect, useState } from "react"; // Loại bỏ useCallback
import { Client } from "@stomp/stompjs";
import SockJS from "sockjs-client";
import type { NetworkBatch } from "../types/ComparisonTypes";

const TOPIC_URL = "/topic/batchpacket";

type ConnectionStatus = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED';

interface UseBatchWebSocketResult {
    receivedBatches: NetworkBatch[];
    connectionStatus: ConnectionStatus;
}

/**
 * Custom Hook để kết nối Websocket và nhận hàng loạt lô gói tin (NetworkBatch).
 */
export const useBatchWebSocket = ( url: string ): UseBatchWebSocketResult => {
    const [ receivedBatches, setReceivedBatches ] = useState<NetworkBatch[]>( [] );
    const [ connectionStatus, setConnectionStatus ] = useState<ConnectionStatus>( 'DISCONNECTED' );

    useEffect( () => {
        // Khai báo hàm xử lý tin nhắn ngay trong useEffect
        const onMessageReceived = ( msg: any ) => {
            try {
                const body: NetworkBatch = JSON.parse( msg.body );
                // Sử dụng hàm cập nhật trạng thái (setReceivedBatches) để tránh dependency phức tạp
                setReceivedBatches( ( prev ) => [ body, ...prev ] );
                console.log( "✅ Received new batch:", body.batchId );
            } catch ( error ) {
                console.error( "❌ Error parsing message body:", error );
            }
        };

        setConnectionStatus('CONNECTING');
        
        const socket = new SockJS( url );
        const client = new Client( {
            webSocketFactory: () => socket as unknown as WebSocket,
            
            onConnect: () => {
                setConnectionStatus( 'CONNECTED' );
                console.log( "✅ Connected to WebSocket at:", url );
                // Chỉ subscribe khi kết nối thành công
                client.subscribe( TOPIC_URL, onMessageReceived );
            },
            
            onStompError: ( err ) => {
                // Thường lỗi STOMP có nghĩa là phải ngắt kết nối
                setConnectionStatus( 'DISCONNECTED' ); 
                console.error( "STOMP error:", err );
            },
            
            onWebSocketClose: () => {
                // Đảm bảo chỉ set DISCONNECTED khi đóng kết nối không phải do logic của chúng ta
                if (client.connected) {
                    // Nếu client vẫn còn connected trong logic, thì đây là lỗi ngoài ý muốn
                    console.log("🔌 Unexpected WebSocket close.");
                }
                setConnectionStatus( 'DISCONNECTED' );
                console.log( "🔌 WebSocket closed." );
            },
            
            debug: (str) => { 
                console.log("STOMP DEBUG:", str); 
            },
        } );

        client.activate();

        return () => {
            // Cleanup: Đảm bảo client được deactivate đúng cách
            if ( client.connected ) {
                client.deactivate();
                console.log( "🔌 STOMP client deactivated." );
            }
            // Nếu client chưa kịp kết nối nhưng đã bị hủy, đảm bảo trạng thái reset
            setConnectionStatus('DISCONNECTED'); 
        };
    }, [ url ] ); // Loại bỏ onMessageReceived khỏi dependency array

    return { receivedBatches, connectionStatus };
};
package com.example.service;

import com.example.model.Packet;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public class PacketReceiver {

    private final ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
    private ServerSocket serverSocket;
    private volatile boolean running = false;
    
    // Maximum packet size (16KB) - must match server
    private static final int MAX_PACKET_SIZE = 16 * 1024;

    // 1. Chỉ tạo MỘT thread pool cho acceptor
    private final ExecutorService acceptorPool = Executors.newSingleThreadExecutor();
    
    // 2. Chỉ tạo MỘT thread pool cho TẤT CẢ client
    private final ExecutorService clientHandlerPool = Executors.newCachedThreadPool();

    /**
     * Start listening...
     */
    public void start(int port, Consumer<Packet> packetConsumer) throws IOException {
        if (running) return;
        serverSocket = new ServerSocket(port);
        running = true;

        acceptorPool.submit(() -> {
            while (running && !serverSocket.isClosed()) {
                try {
                    Socket client = serverSocket.accept();
                    // 3. Dùng chung pool 'clientHandlerPool'
                    handleClient(client, packetConsumer);
                } catch (IOException e) {
                    if (running) {
                        System.err.println("Acceptor thread error: " + e.getMessage());
                    }
                }
            }
        });
    }

    /**
     * Handle client connection using length-prefix protocol (matching server)
     * Protocol: [4-byte integer length N][N bytes of JSON data]
     */
    private void handleClient(Socket client, Consumer<Packet> packetConsumer) {
        // 4. Gửi tác vụ xử lý client vào pool chung
        clientHandlerPool.submit(() -> {
            // 5. Đưa Socket vào try-with-resources để nó TỰ ĐỘNG đóng
            try (Socket clientSocket = client; 
                 DataInputStream dis = new DataInputStream(clientSocket.getInputStream())) {
                
                System.out.println("✅ Client connected from: " + clientSocket.getRemoteSocketAddress());
                
                // 6. Vòng lặp: Đọc liên tục cho đến khi client ngắt kết nối
                while (running) {
                    int packetLength;
                    byte[] lengthBytes = new byte[4];
                    
                    try {
                        // Read 4-byte length prefix
                        dis.readFully(lengthBytes);
                        
                        // Convert to int using big-endian format (network byte order)
                        packetLength = ((lengthBytes[0] & 0xFF) << 24) |
                                       ((lengthBytes[1] & 0xFF) << 16) |
                                       ((lengthBytes[2] & 0xFF) << 8) |
                                       (lengthBytes[3] & 0xFF);
                    } catch (EOFException eof) {
                        // Client closed connection cleanly
                        System.out.println("📤 Client disconnected: " + clientSocket.getRemoteSocketAddress());
                        break;
                    }
                    
                    // Validate packet length
                    if (packetLength <= 0 || packetLength > MAX_PACKET_SIZE) {
                        System.err.println("❌ Invalid packet length: " + packetLength + 
                            " (expected 1-" + MAX_PACKET_SIZE + " bytes)");
                        break;
                    }
                    
                    // Read exactly packetLength bytes for the payload
                    byte[] data = new byte[packetLength];
                    dis.readFully(data);
                    
                    try {
                        // Deserialize JSON packet
                        Packet p = mapper.readValue(data, Packet.class);
                        System.out.println("📩 Received packet: " + p.getPacketId() + 
                            " from " + p.getSourceUserId() + " (" + data.length + " bytes)");
                        packetConsumer.accept(p); // Gửi packet đến người nghe
                    } catch (Exception ex) {
                        System.err.println("❌ Failed to parse packet JSON: " + ex.getMessage());
                    }
                }
                
            } catch (SocketException e) {
                // Connection reset by peer - normal when client disconnects
                System.out.println("🔌 Client disconnected: " + e.getMessage());
            } catch (IOException e) {
                System.err.println("❌ I/O error handling client: " + e.getMessage());
                e.printStackTrace();
            }
            // Socket sẽ tự động đóng ở đây
        });
    }

    /**
     * Stop listening...
     */
    public void stop() {
        running = false;
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) {}
        
        // 7. Tắt CẢ HAI thread pool
        acceptorPool.shutdownNow();
        clientHandlerPool.shutdownNow();
    }
}
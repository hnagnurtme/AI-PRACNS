package com.example.service;

import com.example.model.Packet;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public class PacketReceiver {

    private final ObjectMapper mapper = new ObjectMapper();
    private ServerSocket serverSocket;
    private volatile boolean running = false;

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

    private void handleClient(Socket client, Consumer<Packet> packetConsumer) {
        // 4. Gửi tác vụ xử lý client vào pool chung
        clientHandlerPool.submit(() -> {
            // 5. Đưa Socket vào try-with-resources để nó TỰ ĐỘNG đóng
            try (Socket clientSocket = client; 
                 BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {
                
                String line;
                // 6. Vòng lặp: Đọc liên tục cho đến khi client ngắt kết nối
                while (running && (line = in.readLine()) != null) {
                    if (line.isEmpty()) continue;

                    try {
                        Packet p = mapper.readValue(line, Packet.class);
                        packetConsumer.accept(p); // Gửi packet đến người nghe
                    } catch (Exception ex) {
                        System.err.println("Failed to parse packet JSON: " + line);
                    }
                }
                
            } catch (IOException e) {
                // Thường là lỗi "Connection reset" khi client ngắt kết nối đột ngột
                // Đây là lỗi bình thường, không cần in stack trace
                System.out.println("Client disconnected: " + e.getMessage());
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
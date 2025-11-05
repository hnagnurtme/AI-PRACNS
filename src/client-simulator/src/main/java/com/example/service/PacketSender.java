package com.example.service;

import com.example.model.Packet;
import com.example.util.PacketSerializerHelper;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Optimized TCP packet sender that caches and reuses connections.
 * This class is a thread-safe singleton and manages its own lifecycle
 * using a JVM shutdown hook to close all connections.
 */
public class PacketSender {

    // 1. Singleton Instance
    private static final PacketSender INSTANCE = new PacketSender();

    private final Map<String, PrintWriter> activeWriters = new ConcurrentHashMap<>();
    private final Map<String, Socket> activeSockets = new ConcurrentHashMap<>();

    /**
     * Private constructor to enforce singleton pattern.
     * Registers a shutdown hook to clean up connections on JVM exit.
     */
    private PacketSender() {
        // 2. Thay thế cho @PreDestroy
        // Đăng ký một "shutdown hook" để tự động gọi closeAll() khi app tắt
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutdown hook running: Closing all active PacketSender connections...");
            this.closeAll();
            System.out.println("All connections closed.");
        }));
    }

    /**
     * Lấy instance duy nhất của PacketSender.
     */
    public static PacketSender getInstance() {
        return INSTANCE;
    }

    /**
     * Send a packet to the given host:port.
     * Reuses an existing connection or creates a new one.
     *
     * @throws IOException when serialization fails or sending fails.
     */
    public void send(String host, int port, Packet packet) throws IOException {
        // Validate input
        if (host == null || host.isBlank()) {
            throw new IOException("Host không được để trống");
        }
        if (port <= 0 || port > 65535) {
            throw new IOException("Port không hợp lệ: " + port);
        }
        if (packet == null) {
            throw new IOException("Packet không được null");
        }
        
        String json = PacketSerializerHelper.serialize(packet);
        if (json == null) {
            throw new IOException("Failed to serialize packet with ID: " + packet.getPacketId());
        }

        String connectionKey = host + ":" + port;

        try {
            PrintWriter writer = getOrCreateWriter(connectionKey, host, port);
            
            // Dùng synchronized trên writer để đảm bảo 
            // 2 thread không ghi đè dữ liệu của nhau trên CÙNG MỘT socket
            synchronized (writer) {
                writer.println(json);
                writer.flush(); // Đảm bảo dữ liệu được gửi ngay
                
                // Kiểm tra lỗi ngay lập tức
                if (writer.checkError()) {
                    throw new IOException("PrintWriter reported an error while sending to " + connectionKey);
                }
            }

        } catch (IOException e) {
            // Nếu có lỗi, xóa kết nối hỏng để lần sau tạo lại
            System.err.println("❌ Connection failed for " + connectionKey + ". Evicting cache. Error: " + e.getMessage());
            closeAndRemoveConnection(connectionKey);
            throw new IOException("Failed to send packet to " + connectionKey + ": " + e.getMessage(), e);
        }
    }

    private PrintWriter getOrCreateWriter(String key, String host, int port) throws IOException {
        // Lần 1: Lấy nhanh (đã có kết nối)
        PrintWriter writer = activeWriters.get(key);
        if (writer != null) {
            Socket socket = activeSockets.get(key);
            // Kiểm tra socket còn mở không
            if (socket != null && !socket.isClosed() && socket.isConnected()) {
                return writer;
            } else {
                // Socket đã đóng, xóa và tạo lại
                System.out.println("⚠️ Detected closed socket for " + key + ", recreating...");
                closeAndRemoveConnection(key);
            }
        }

        // Lần 2: Nếu không có, phải khóa lại để tạo mới
        // Dùng 'this' để khóa toàn bộ object PacketSender
        synchronized (this) {
            // Kiểm tra lại (double-checked locking)
            writer = activeWriters.get(key);
            if (writer != null) {
                Socket socket = activeSockets.get(key);
                if (socket != null && !socket.isClosed() && socket.isConnected()) {
                    return writer;
                }
            }

            // Tạo kết nối mới
            System.out.println("🔌 Creating new persistent connection to " + key);
            try {
                Socket socket = new Socket();
                socket.connect(new java.net.InetSocketAddress(host, port), 5000); // 5s timeout
                socket.setSoTimeout(5000); // Read timeout
                socket.setKeepAlive(true); // Giữ kết nối
                
                PrintWriter newWriter = new PrintWriter(socket.getOutputStream(), true); // 'true' = autoFlush

                activeSockets.put(key, socket);
                activeWriters.put(key, newWriter);
                
                System.out.println("✅ Successfully connected to " + key);
                return newWriter;
            } catch (IOException e) {
                System.err.println("❌ Failed to create connection to " + key + ": " + e.getMessage());
                throw new IOException("Cannot connect to " + key + ": " + e.getMessage(), e);
            }
        }
    }

    private void closeAndRemoveConnection(String key) {
        // Khóa lại để đảm bảo thread-safety khi xóa
        synchronized (this) {
            activeWriters.remove(key);
            
            Socket socket = activeSockets.remove(key);
            if (socket != null) {
                try {
                    socket.close();
                } catch (IOException e) {
                    // Bỏ qua lỗi khi đóng
                }
            }
        }
    }

    /**
     * Đóng tất cả các kết nối đang hoạt động.
     * Hàm này chủ yếu được gọi bởi shutdown hook.
     */
    public void closeAll() {
        // Dùng .keySet() để tránh ConcurrentModificationException
        for (String key : activeSockets.keySet()) {
            closeAndRemoveConnection(key);
        }
    }
}
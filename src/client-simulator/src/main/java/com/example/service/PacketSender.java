package com.example.service;

import com.example.model.Packet;
import com.example.util.PacketSerializerHelper;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Optimized TCP packet sender that caches and reuses connections.
 * This class is a thread-safe singleton and manages its own lifecycle
 * using a JVM shutdown hook to close all connections.
 *
 * It uses a length-prefix protocol:
 * - 4-byte Big-Endian integer for the length of the payload.
 * - UTF-8 encoded JSON payload.
 */
public class PacketSender {

    // 1. Singleton Instance
    private static final PacketSender INSTANCE = new PacketSender();

    private final Map<String, OutputStream> activeStreams = new ConcurrentHashMap<>();
    private final Map<String, Socket> activeSockets = new ConcurrentHashMap<>();

    /**
     * Private constructor to enforce singleton pattern.
     * Registers a shutdown hook to clean up connections on JVM exit.
     */
    private PacketSender() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutdown hook running: Closing all active PacketSender connections...");
            this.closeAll();
            System.out.println("All connections closed.");
        }));
    }

    /**
     * Gets the singleton instance of PacketSender.
     */
    public static PacketSender getInstance() {
        return INSTANCE;
    }

    /**
     * Converts an integer to a 4-byte array in big-endian format (network byte order).
     */
    private byte[] intToBytes(int value) {
        return new byte[] {
            (byte) (value >> 24),
            (byte) (value >> 16),
            (byte) (value >> 8),
            (byte) value
        };
    }

    /**
     * Send a packet to the given host:port using a length-prefix protocol.
     * Reuses an existing connection or creates a new one.
     *
     * @throws IOException when serialization fails or sending fails.
     */
    public void send(String host, int port, Packet packet) throws IOException {
        // Validate input
        if (host == null || host.isBlank()) {
            throw new IOException("Host cannot be empty");
        }
        if (port <= 0 || port > 65535) {
            throw new IOException("Invalid port: " + port);
        }
        if (packet == null) {
            throw new IOException("Packet cannot be null");
        }
        
        String json = PacketSerializerHelper.serialize(packet);
        if (json == null) {
            throw new IOException("Failed to serialize packet with ID: " + packet.getPacketId());
        }

        byte[] payload = json.getBytes(StandardCharsets.UTF_8);
        byte[] lengthPrefix = intToBytes(payload.length);

        String connectionKey = host + ":" + port;

        try {
            OutputStream out = getOrCreateStream(connectionKey, host, port);
            
            // Synchronize on the stream to ensure thread-safe writing on the same socket
            synchronized (out) {
                out.write(lengthPrefix);
                out.write(payload);
                out.flush(); // Ensure data is sent immediately
            }

        } catch (IOException e) {
            // If an error occurs, evict the faulty connection from the cache
            System.err.println("❌ Connection failed for " + connectionKey + ". Evicting cache. Error: " + e.getMessage());
            closeAndRemoveConnection(connectionKey);
            throw new IOException("Failed to send packet to " + connectionKey + ": " + e.getMessage(), e);
        }
    }

    private OutputStream getOrCreateStream(String key, String host, int port) throws IOException {
        // First check: quick retrieval for existing connections
        OutputStream out = activeStreams.get(key);
        if (out != null) {
            Socket socket = activeSockets.get(key);
            // Validate the underlying socket is still open and connected
            if (socket != null && !socket.isClosed() && socket.isConnected()) {
                return out;
            } else {
                // Socket is closed, evict and recreate
                System.out.println("⚠️ Detected closed socket for " + key + ", recreating...");
                closeAndRemoveConnection(key);
            }
        }

        // Second check: synchronized block for creating a new connection
        synchronized (this) {
            // Double-checked locking
            out = activeStreams.get(key);
            if (out != null) {
                Socket socket = activeSockets.get(key);
                if (socket != null && !socket.isClosed() && socket.isConnected()) {
                    return out;
                }
            }

            // Create a new connection
            System.out.println("🔌 Creating new persistent connection to " + key);
            try {
                Socket socket = new Socket();
                socket.connect(new InetSocketAddress(host, port), 5000); // 5s connect timeout
                socket.setSoTimeout(5000); // 5s read timeout
                socket.setKeepAlive(true);
                
                OutputStream newStream = socket.getOutputStream();

                activeSockets.put(key, socket);
                activeStreams.put(key, newStream);
                
                System.out.println("✅ Successfully connected to " + key);
                return newStream;
            } catch (IOException e) {
                System.err.println("❌ Failed to create connection to " + key + ": " + e.getMessage());
                throw new IOException("Cannot connect to " + key + ": " + e.getMessage(), e);
            }
        }
    }

    private void closeAndRemoveConnection(String key) {
        // Synchronize to ensure thread-safety during removal
        synchronized (this) {
            activeStreams.remove(key);
            
            Socket socket = activeSockets.remove(key);
            if (socket != null) {
                try {
                    socket.close();
                } catch (IOException e) {
                    // Ignore errors on close
                }
            }
        }
    }

    /**
     * Closes all active connections.
     * This is primarily called by the shutdown hook.
     */
    public void closeAll() {
        // Use keySet to avoid ConcurrentModificationException
    for (String key : activeSockets.keySet()) {
            closeAndRemoveConnection(key);
        }
    }
}
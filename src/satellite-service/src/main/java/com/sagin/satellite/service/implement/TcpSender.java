package com.sagin.satellite.service.implement;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.common.SatelliteException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * TcpSender gửi packet qua TCP connection đến các node khác
 * Hỗ trợ connection pooling, retry logic, và metrics tracking
 */
public class TcpSender {
    private static final Logger logger = LoggerFactory.getLogger(TcpSender.class);

    // Configuration
    private final ObjectMapper mapper = new ObjectMapper();
    private final int sendTimeoutMs;
    private final int connectionTimeoutMs;
    private final boolean enableConnectionReuse;

    // Address resolution cache
    private final Map<String, String> nodeAddressCache = new ConcurrentHashMap<>();

    // Connection pool (simple implementation)
    private final Map<String, Socket> connectionPool = new ConcurrentHashMap<>();

    // Metrics
    private final AtomicLong totalPacketsSent = new AtomicLong(0);
    private final AtomicLong totalSendFailures = new AtomicLong(0);
    private final AtomicLong totalConnectionFailures = new AtomicLong(0);

    /**
     * Constructor với cấu hình mặc định
     */
    public TcpSender() {
        this(2000, 1000, true);
    }

    /**
     * Constructor với cấu hình tùy chỉnh
     */
    public TcpSender(int sendTimeoutMs, int connectionTimeoutMs, boolean enableConnectionReuse) {
        this.sendTimeoutMs = sendTimeoutMs;
        this.connectionTimeoutMs = connectionTimeoutMs;
        this.enableConnectionReuse = enableConnectionReuse;

        logger.info("TcpSender initialized: sendTimeout={}ms, connectionTimeout={}ms, reuseConnections={}",
                sendTimeoutMs, connectionTimeoutMs, enableConnectionReuse);
    }

    /**
     * Gửi packet đến destination node
     */
    public void send(Packet packet) throws Exception {
        validatePacket(packet);

        String destination = resolveDestination(packet);
        if (destination == null) {
            totalSendFailures.incrementAndGet();
            throw new SatelliteException.SendException(
                    "Cannot resolve destination for packet: " + packet.getPacketId());
        }

        Socket socket = null;
        try {
            socket = getConnection(destination);
            sendPacketData(socket, packet);
            totalPacketsSent.incrementAndGet();

            logger.debug("Packet {} sent successfully to {}", packet.getPacketId(), destination);

        } catch (Exception e) {
            totalSendFailures.incrementAndGet();
            // Đóng connection bị lỗi
            if (socket != null && !socket.isClosed()) {
                closeConnection(destination);
            }
            throw new SatelliteException.SendException(
                    "Failed to send packet " + packet.getPacketId() + " to " + destination);
        } finally {
            // Nếu không reuse connection, đóng ngay
            if (!enableConnectionReuse && socket != null) {
                closeConnection(destination);
            }
        }
    }

    /**
     * Validate packet trước khi gửi
     */
    private void validatePacket(Packet packet) throws SatelliteException.InvalidPacketException {
        if (packet == null) {
            throw new SatelliteException.InvalidPacketException("Packet cannot be null");
        }
        if (packet.getPacketId() == null || packet.getPacketId().trim().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Packet ID cannot be null or empty");
        }
        if (packet.getNextHop() == null || packet.getNextHop().trim().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Next hop cannot be null or empty");
        }
        if (packet.getTTL() <= 0) {
            throw new SatelliteException.InvalidPacketException("TTL must be positive");
        }
        if (packet.isDropped()) {
            throw new SatelliteException.InvalidPacketException("Cannot send dropped packet");
        }
    }

    /**
     * Resolve destination address từ next hop
     */
    private String resolveDestination(Packet packet) {
        String nextHop = packet.getNextHop();

        // Check cache first
        String cachedAddress = nodeAddressCache.get(nextHop);
        if (cachedAddress != null) {
            return cachedAddress;
        }

        // Resolve address (có thể từ DNS, service discovery, config file, etc.)
        String resolvedAddress = resolveNodeAddress(nextHop);
        if (resolvedAddress != null) {
            nodeAddressCache.put(nextHop, resolvedAddress);
        }

        return resolvedAddress;
    }

    /**
     * Resolve node address - có thể override để implement custom resolution
     */
    protected String resolveNodeAddress(String nodeId) {
        String serverName = System.getenv("SERVER_HOST");
        String targetPort = System.getenv("SERVER_TARGET_PORT");
        if (serverName == null)
            serverName = "server8000";
        if (targetPort == null)
            targetPort = "8000";

        switch (nodeId) {
            case "ground-station-1":
                return serverName + ":" + targetPort;
            default:
                return null;
        }
    }

    private Socket getConnection(String destination) throws Exception {
        Socket socket = null;

        if (enableConnectionReuse) {
            socket = connectionPool.get(destination);
            if (socket != null && (socket.isClosed() || !socket.isConnected())) {
                // Connection không hợp lệ, remove khỏi pool
                connectionPool.remove(destination);
                socket = null;
            }
        }

        if (socket == null) {
            socket = createConnection(destination);
            if (enableConnectionReuse) {
                connectionPool.put(destination, socket);
            }
        }

        return socket;
    }

    /**
     * Tạo TCP connection mới
     */
    private Socket createConnection(String destination) throws Exception {
        try {
            String[] parts = destination.split(":");
            if (parts.length != 2) {
                throw new IllegalArgumentException("Invalid destination format: " + destination);
            }

            String host = parts[0];
            int port = Integer.parseInt(parts[1]);

            Socket socket = new Socket();
            socket.setSoTimeout(sendTimeoutMs);
            socket.setKeepAlive(true);
            socket.setTcpNoDelay(true);

            // Connect với timeout
            socket.connect(new InetSocketAddress(host, port), connectionTimeoutMs);

            logger.debug("TCP connection established to {}", destination);
            return socket;

        } catch (Exception e) {
            totalConnectionFailures.incrementAndGet();
            logger.error("Failed to create connection to {}: {}", destination, e.getMessage());
            throw e;
        }
    }

    private void sendPacketData(Socket socket, Packet packet) {
        try {
            packet.setNextHop("STATION-2");
            OutputStream out = socket.getOutputStream();
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8));

            // Serialize packet sang JSON
            String jsonPacket = mapper.writeValueAsString(packet);

            // Gửi kèm \n để server đọc theo dòng
            writer.write(jsonPacket);
            writer.newLine();
            writer.flush();

            logger.debug("Sent {} chars for packet {}", jsonPacket.length(), packet.getPacketId());

        } catch (SocketTimeoutException e) {
            logger.error("Send timeout for packet {}", packet.getPacketId());
            throw new RuntimeException(e);
        } catch (IOException e) {
            logger.error("IO error sending packet {}: {}", packet.getPacketId(), e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Đóng connection đến destination
     */
    public void closeConnection(String destination) {
        Socket socket = connectionPool.remove(destination);
        if (socket != null) {
            try {
                if (!socket.isClosed()) {
                    socket.close();
                    logger.debug("Closed connection to {}", destination);
                }
            } catch (IOException e) {
                logger.warn("Error closing connection to {}: {}", destination, e.getMessage());
            }
        }
    }

    /**
     * Đóng tất cả connections
     */
    public void closeAllConnections() {
        logger.info("Closing all TCP connections ({} active)", connectionPool.size());

        for (String destination : connectionPool.keySet()) {
            closeConnection(destination);
        }

        connectionPool.clear();
        logger.info("All TCP connections closed");
    }

    /**
     * Thêm hoặc update node address mapping
     */
    public void addNodeMapping(String nodeId, String address) {
        nodeAddressCache.put(nodeId, address);
        logger.info("Added node mapping: {} -> {}", nodeId, address);
    }

    /**
     * Remove node mapping
     */
    public void removeNodeMapping(String nodeId) {
        String removed = nodeAddressCache.remove(nodeId);
        if (removed != null) {
            logger.info("Removed node mapping: {} -> {}", nodeId, removed);
            // Đóng connection nếu có
            closeConnection(removed);
        }
    }

    /**
     * Lấy metrics
     */
    public TcpSenderMetrics getMetrics() {
        return new TcpSenderMetrics(
                totalPacketsSent.get(),
                totalSendFailures.get(),
                totalConnectionFailures.get(),
                connectionPool.size(),
                nodeAddressCache.size());
    }

    /**
     * Health check
     */
    public boolean isHealthy() {
        // Simple health check - có thể mở rộng
        double failureRate = totalPacketsSent.get() > 0
                ? (double) totalSendFailures.get() / (totalPacketsSent.get() + totalSendFailures.get())
                : 0.0;

        return failureRate < 0.1; // < 10% failure rate
    }

    /**
     * Cleanup resources
     */
    public void shutdown() {
        logger.info("TcpSender shutting down...");
        closeAllConnections();
        nodeAddressCache.clear();
        logger.info("TcpSender shutdown completed. Final metrics: {}", getMetrics());
    }

    /**
     * Metrics class cho TcpSender
     */
    public static class TcpSenderMetrics {
        private final long totalPacketsSent;
        private final long totalSendFailures;
        private final long totalConnectionFailures;
        private final int activeConnections;
        private final int cachedNodes;

        public TcpSenderMetrics(long totalPacketsSent, long totalSendFailures,
                long totalConnectionFailures, int activeConnections, int cachedNodes) {
            this.totalPacketsSent = totalPacketsSent;
            this.totalSendFailures = totalSendFailures;
            this.totalConnectionFailures = totalConnectionFailures;
            this.activeConnections = activeConnections;
            this.cachedNodes = cachedNodes;
        }

        public long getTotalPacketsSent() {
            return totalPacketsSent;
        }

        public long getTotalSendFailures() {
            return totalSendFailures;
        }

        public long getTotalConnectionFailures() {
            return totalConnectionFailures;
        }

        public int getActiveConnections() {
            return activeConnections;
        }

        public int getCachedNodes() {
            return cachedNodes;
        }

        public double getSuccessRate() {
            long total = totalPacketsSent + totalSendFailures;
            return total > 0 ? (double) totalPacketsSent / total : 1.0;
        }

        @Override
        public String toString() {
            return String.format("TcpSenderMetrics{sent=%d, failures=%d, connFailures=%d, " +
                    "activeConns=%d, cachedNodes=%d, successRate=%.2f%%}",
                    totalPacketsSent, totalSendFailures, totalConnectionFailures,
                    activeConnections, cachedNodes, getSuccessRate() * 100);
        }
    }
}


package com.sagin.satellite.service.implement;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sagin.satellite.model.Packet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class TcpSender {
    private static final Logger logger = LoggerFactory.getLogger(TcpSender.class);
    private final ObjectMapper mapper = new ObjectMapper();
    private final ManagerClient managerClient;
    private final int sendTimeoutMs = 2000;
    private final Map<String, String> nodeAddressCache = new ConcurrentHashMap<>();

    public TcpSender(ManagerClient managerClient) {
        this.managerClient = managerClient;
    }

    public void send(Packet packet) throws Exception {
        String nextHop = packet.getNextHop();
        if (nextHop == null) throw new IllegalArgumentException("nextHop missing");

        String address = nodeAddressCache.computeIfAbsent(nextHop, 
            k -> managerClient.getNodeAddress(k));
        if (address == null) {
            logger.warn("Unknown nextHop address for {}", nextHop);
            throw new IllegalStateException("Missing node address");
        }
        final String[] parts = address.split(":");
        final String ip = parts[0];
        final int port = Integer.parseInt(parts[1]);

        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(ip, port), sendTimeoutMs);
            OutputStream out = socket.getOutputStream();
            String json = mapper.writeValueAsString(packet);
            out.write(json.getBytes(StandardCharsets.UTF_8));
            out.write('\n');
            out.flush();
            logger.debug("Sent packet {} to {}:{}", packet.getPacketId(), ip, port);
        } catch (Exception ex) {
            logger.error("Failed to send packet {} to {}: {}", packet.getPacketId(), address, ex.getMessage());
            throw ex;
        }
    }
}

package com.sagin;

import com.sagin.factory.QoSProfileFactory;
import com.sagin.model.Packet;
import com.sagin.model.ServiceType;
import com.sagin.util.AppLogger;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.slf4j.Logger;

import java.io.OutputStream;
import java.net.Socket;
import java.net.InetSocketAddress;
import java.util.ArrayList;

public class SimulationClient {
    private static final Logger logger = AppLogger.getLogger(SimulationClient.class);

    // Hàm chuyển int -> 4 byte big-endian (length prefix)
    private static byte[] intToBytes(int value) {
        return new byte[]{
                (byte) (value >> 24),
                (byte) (value >> 16),
                (byte) (value >> 8),
                (byte) value
        };
    }

    public static void main(String[] args) {
        String host = "localhost";
        int port = 7765;

        String requestId = "REQ-" + System.currentTimeMillis();
        AppLogger.putMdc("requestId", requestId);
        logger.info("Starting simulation client, sending packet to {}:{}", host, port);

        // --- 1. Tạo packet mẫu ---
        Packet packet = new Packet();
        packet.setPacketId("P-TEST-001");
        packet.setType("TEST");
        packet.setPayloadDataBase64("SGVsbG8gV29ybGQ="); // "Hello World" base64
        packet.setPayloadSizeByte(12012);
        packet.setSourceUserId("user-01");
        packet.setDestinationUserId("user-Singapore");
        packet.setStationSource("GS_HANOI");
        packet.setStationDest("GS_SINGAPORE");
        packet.setTTL(40);
        packet.setPriorityLevel(1);
        packet.setMaxAcceptableLatencyMs(500);
        packet.setMaxAcceptableLossRate(0.01);
        packet.setServiceQoS(QoSProfileFactory.getQosProfile(ServiceType.AUDIO_CALL));
        packet.setHopRecords(new ArrayList<>());
        packet.setPathHistory(new ArrayList<>());
        packet.setTimeSentFromSourceMs(System.currentTimeMillis());
        packet.setAcknowledgedPacketId(null);
        packet.setCurrentHoldingNodeId(packet.getStationSource());
        packet.setNextHopNodeId(null);
        packet.setDropped(false);
        packet.setDropReason(null);
        packet.setAccumulatedDelayMs(0);
        packet.setAnalysisData(null);
        packet.setUseRL(true);

        // --- 2. Chuẩn bị ObjectMapper ---
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());

        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(host, port), 1000); // timeout 1s
            try (OutputStream os = socket.getOutputStream()) {

                // Chuyển packet thành JSON UTF-8
                byte[] packetBytes = objectMapper.writeValueAsBytes(packet);

                // --- Thêm 4-byte length prefix ---
                byte[] lengthPrefix = intToBytes(packetBytes.length);
                os.write(lengthPrefix);   // gửi độ dài trước
                os.write(packetBytes);    // gửi dữ liệu JSON
                os.flush();

                logger.info("✅ Packet sent successfully to NodeGateway (packet ID: {})", packet.getPacketId());
            }
        } catch (Exception e) {
            logger.error("Failed to send packet to NodeGateway at {}:{}", host, port, e);
        } finally {
            AppLogger.clearMdc();
        }
    }
}

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
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

public class SimulationClient {
    private static final Logger logger = AppLogger.getLogger(SimulationClient.class);
    
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
        packet.setDestinationUserId("user-02");
        packet.setStationSource("GS_HANOI");
        packet.setStationDest("GS_DANANG");
        packet.setTTL(40);
        packet.setPriorityLevel(1);
        packet.setMaxAcceptableLatencyMs(500);
        packet.setMaxAcceptableLossRate(0.01);
        packet.setServiceQoS(QoSProfileFactory.getQosProfile(ServiceType.AUDIO_CALL));
        packet.setHopRecords(new ArrayList<>());
        packet.setPathHistory(new ArrayList<>());
        packet.setTimeSentFromSourceMs(System.currentTimeMillis());
        packet.setAcknowledgedPacketId(null);
        packet.setCurrentHoldingNodeId(packet.getStationSource()); // ⚡ quan trọng
        packet.setNextHopNodeId(null);
        packet.setDropped(false);
        packet.setDropReason(null);
        packet.setAccumulatedDelayMs(0);
        packet.setAnalysisData(null);
        packet.setUseRL(true);

        // --- 2. Chuẩn bị ObjectMapper ---
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());

        try (Socket socket = new Socket(host, port);
             OutputStream os = socket.getOutputStream()) {

            // Chuyển packet thành JSON UTF-8
            String jsonPacket = objectMapper.writeValueAsString(packet);
            logger.debug("Sending packet JSON: {}", jsonPacket);

            os.write(jsonPacket.getBytes(StandardCharsets.UTF_8));
            os.flush();

            logger.info("✅ Packet sent successfully to NodeGateway (packet ID: {})", packet.getPacketId());

        } catch (Exception e) {
            logger.error("Failed to send packet to NodeGateway at {}:{}", host, port, e);
        } finally {
            AppLogger.clearMdc();
        }
    }
}

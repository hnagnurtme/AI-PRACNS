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
import java.util.UUID;

/**
 * Client gửi CẶP packets (Dijkstra + RL) CÙNG LÚC để so sánh công bằng
 */
public class SimulationPairClient {
    private static final Logger logger = AppLogger.getLogger(SimulationPairClient.class);
    
    public static void main(String[] args) {
        String host = "localhost";
        int port = 7765; // Port của GS_HANOI hoặc GS khác
        
        String requestId = "REQ-" + System.currentTimeMillis();
        AppLogger.putMdc("requestId", requestId);
        logger.info("Starting PAIR simulation client, sending 2 packets to {}:{}", host, port);

        // Timestamp chung cho cả 2 packets
        long sharedTimestamp = System.currentTimeMillis();
        String basePacketId = UUID.randomUUID().toString();

        // --- 1. Tạo packet Dijkstra ---
        Packet dijkstraPacket = createPacket(basePacketId, sharedTimestamp, false);
        
        // --- 2. Tạo packet RL ---
        Packet rlPacket = createPacket(basePacketId, sharedTimestamp, true);

        // --- 3. ObjectMapper ---
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());

        // --- 4. Gửi 2 packets ĐỒNG THỜI (parallel threads) ---
        Thread dijkstraThread = new Thread(() -> sendPacket(dijkstraPacket, host, port, objectMapper, "Dijkstra"));
        Thread rlThread = new Thread(() -> sendPacket(rlPacket, host, port, objectMapper, "RL"));
        
        // Start cùng lúc
        dijkstraThread.start();
        rlThread.start();
        
        // Đợi cả 2 hoàn thành
        try {
            dijkstraThread.join();
            rlThread.join();
            logger.info("✅ Both packets sent successfully!");
        } catch (InterruptedException e) {
            logger.error("Failed to send packets", e);
        } finally {
            AppLogger.clearMdc();
        }
    }
    
    /**
     * Tạo packet với base ID và timestamp chung
     */
    private static Packet createPacket(String baseId, long timestamp, boolean useRL) {
        Packet packet = new Packet();
        packet.setPacketId(baseId); // ✅ Cùng base ID
        packet.setType("TEST_COMPARISON");
        packet.setPayloadDataBase64("WElOIENIQU8="); // "XIN CHAO" base64
        packet.setPayloadSizeByte(12);
        packet.setSourceUserId("user-Hue");
        packet.setDestinationUserId("user-DaNang");
        packet.setStationSource("GS_DANANG");
        packet.setStationDest("GS_DANANG");
        packet.setTTL(30);
        packet.setPriorityLevel(2);
        packet.setMaxAcceptableLatencyMs(80.0);
        packet.setMaxAcceptableLossRate(0.005);
        packet.setServiceQoS(QoSProfileFactory.getQosProfile(ServiceType.AUDIO_CALL));
        packet.setHopRecords(new ArrayList<>());
        packet.setPathHistory(new ArrayList<>());
        packet.setTimeSentFromSourceMs(timestamp); // ✅ Cùng timestamp
        packet.setAcknowledgedPacketId(null);
        packet.setCurrentHoldingNodeId(packet.getStationSource());
        packet.setNextHopNodeId(null);
        packet.setDropped(false);
        packet.setDropReason(null);
        packet.setAccumulatedDelayMs(0);
        packet.setAnalysisData(null);
        packet.setUseRL(useRL); // ✅ Khác nhau chỉ ở đây
        
        return packet;
    }
    
    /**
     * Gửi packet qua TCP socket
     */
    private static void sendPacket(Packet packet, String host, int port, ObjectMapper mapper, String label) {
        try (Socket socket = new Socket(host, port);
             OutputStream os = socket.getOutputStream()) {

            String jsonPacket = mapper.writeValueAsString(packet);
            logger.debug("[{}] Sending packet: {}", label, packet.getPacketId());

            os.write(jsonPacket.getBytes(StandardCharsets.UTF_8));
            os.flush();

            logger.info("✅ [{}] Packet sent: {} | useRL={}", 
                    label, packet.getPacketId(), packet.isUseRL());

        } catch (Exception e) {
            logger.error("[{}] Failed to send packet", label, e);
        }
    }
}

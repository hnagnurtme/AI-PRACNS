package com.sagin;

import com.sagin.factory.QoSProfileFactory;
import com.sagin.model.Packet;
import com.sagin.model.ServiceType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

public class SimulationClient {
    public static void main(String[] args) {
        String host = "localhost";
        int port = 5001;
        Packet packet = new Packet();
        packet.setPacketId("P-TEST-001");
        packet.setType("TEST");
        packet.setPayloadDataBase64("SGVsbG8gV29ybGQ="); // "Hello World" base64
        packet.setPayloadSizeByte(11);
        packet.setSourceUserId("user-tokyo");
        packet.setDestinationUserId("user-paris");
        packet.setStationSource("N-HANOI");
        packet.setStationDest("N-NEWYORK");
        packet.setTTL(10);
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
        packet.setUseRL(false);

        // --- 2. Chuẩn bị ObjectMapper ---
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());


        try (Socket socket = new Socket(host, port);
            OutputStream os = socket.getOutputStream()) {

            String jsonPacket = objectMapper.writeValueAsString(packet);
            System.out.println("Đang gửi packet: " + jsonPacket);

            os.write(jsonPacket.getBytes(StandardCharsets.UTF_8));
            os.flush();

            System.out.println("✅ Packet đã gửi tới NodeGateway!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

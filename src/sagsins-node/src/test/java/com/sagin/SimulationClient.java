package com.sagin;

import com.sagin.factory.QoSProfileFactory;
import com.sagin.model.Packet;
import com.sagin.model.ServiceType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.OutputStream;
import java.net.Socket;
import java.util.Collections;

public class SimulationClient {
    public static void main(String[] args) {
        String host = "localhost";
        int port = 8080;

        // Tạo packet mẫu
        Packet packet = new Packet();
        packet.setPacketId("P-TEST-001");
        packet.setType("TEST");
        packet.setPayloadDataBase64("SGVsbG8gV29ybGQ="); // "Hello World" base64
        packet.setSourceUserId("U-001");
        packet.setDestinationUserId("U-002");
        packet.setStationSource("Station-A");
        packet.setStationDest("Station-B");
        packet.setTTL(10);
        packet.setPriorityLevel(1);
        packet.setMaxAcceptableLatencyMs(500);
        packet.setMaxAcceptableLossRate(0.01);
        packet.setServiceQoS(QoSProfileFactory.getQosProfile(ServiceType.AUDIO_CALL));
        packet.setHopRecords(Collections.emptyList());
        packet.setTimeSentFromSourceMs(System.currentTimeMillis());
        packet.setAcknowledgedPacketId(null);
        packet.setCurrentHoldingNodeId(null);
        packet.setPathHistory(Collections.emptyList());
        packet.setDropReason(null);
        packet.setAccumulatedDelayMs(0);
        packet.setDropped(false);
        packet.setPayloadSizeByte(11);
        packet.setAnalysisData(null);
        packet.setUseRL(false);
        packet.setNextHopNodeId(null);

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());

        try (Socket socket = new Socket(host, port);
             OutputStream os = socket.getOutputStream()) {

            // Chuyển packet thành JSON
            String jsonPacket = objectMapper.writeValueAsString(packet);

            os.write(jsonPacket.getBytes());
            os.flush();

            System.out.println("Packet đã gửi tới NodeGateway!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

package com.sagin.service.__test__;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.interfaces.ITCP_Service;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Unit Test cho NodeGateway.
 * Sử dụng real Socket và ObjectMapper thay vì mock để tránh vấn đề với Java 23.
 */
@ExtendWith(MockitoExtension.class)
class NodeGatewayTest {

    @Mock
    private ITCP_Service mockTcpService;

    // Use real ObjectMapper for serialization
    private ObjectMapper objectMapper;
    private NodeGateway nodeGateway;

    private NodeInfo testNodeInfo;
    private String testPacketJson;
    private Packet testPacketObject;

    @BeforeEach
    void setUp() throws IOException {
        // Initialize real ObjectMapper
        objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
        
        // Create NodeGateway with mocked TCP service
        // NodeGateway will create its own ObjectMapper internally
        nodeGateway = new NodeGateway(mockTcpService);
        
        testNodeInfo = new NodeInfo();
        testNodeInfo.setNodeId("GW-Node-1");

        testPacketObject = new Packet();
        testPacketObject.setPacketId("Incoming-P1");
        testPacketObject.setStationSource("Source-Node");
        testPacketObject.setStationDest("GW-Node-1");
        testPacketObject.setTimeSentFromSourceMs(Instant.now().toEpochMilli());
        testPacketObject.setPayloadDataBase64("SGVsbG8=");
        testPacketObject.setPayloadSizeByte(5);

        testPacketJson = objectMapper.writeValueAsString(testPacketObject);
    }

    @Test
    @DisplayName("Test handleClient xử lý thành công packet hợp lệ")
    void testHandleClient_Success() throws Exception {
        // ✅ BUG FIX: Use the correct length-prefixed protocol (4-byte length + data)
        // Create a properly formatted packet with length prefix
        byte[] jsonBytes = testPacketJson.getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(jsonBytes.length); // Write 4-byte length prefix
        dos.write(jsonBytes); // Write actual JSON data
        dos.flush();
        byte[] packetWithLength = baos.toByteArray();
        
        // Find an available port and start listening
        ServerSocket tempSocket = new ServerSocket(0);
        int port = tempSocket.getLocalPort();
        tempSocket.close(); // Close it so nodeGateway can bind to it
        
        // Give a brief moment for the OS to release the port
        Thread.sleep(100);
        
        // Start listening on the port
        nodeGateway.startListening(testNodeInfo, port);
        
        // Give time for the gateway to start
        Thread.sleep(200);
        
        // Connect as client and send data
        try (Socket clientSocket = new Socket("localhost", port);
             OutputStream out = clientSocket.getOutputStream()) {
            out.write(packetWithLength);
            out.flush();
        }

        // Verify
        ArgumentCaptor<Packet> packetCaptor = ArgumentCaptor.forClass(Packet.class);
        verify(mockTcpService, timeout(2000).times(1)).receivePacket(packetCaptor.capture());

        Packet capturedPacket = packetCaptor.getValue();
        assertNotNull(capturedPacket);
        assertEquals(testPacketObject.getPacketId(), capturedPacket.getPacketId());
        assertEquals(testNodeInfo.getNodeId(), capturedPacket.getCurrentHoldingNodeId());
        
        // Clean up
        nodeGateway.stopListening();
    }

    @Test
    @DisplayName("Test handleClient xử lý lỗi khi JSON không hợp lệ")
    void testHandleClient_InvalidJson() throws Exception {
        String invalidJson = "{\"packetId\":\"Bad\", \"stationSource\":";
        
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            int port = serverSocket.getLocalPort();
            
            Thread serverThread = new Thread(() -> {
                try {
                    Socket clientSocket = serverSocket.accept();
                    nodeGateway.startListening(testNodeInfo, port);
                    nodeGateway.handleClient(clientSocket);
                } catch (Exception e) {
                    // Expected
                }
            });
            serverThread.start();

            try (Socket clientSocket = new Socket("localhost", port);
                 OutputStream out = clientSocket.getOutputStream()) {
                out.write(invalidJson.getBytes(StandardCharsets.UTF_8));
                out.flush();
            }

            serverThread.join(2000);

            // Verify no packet was received
            verify(mockTcpService, after(500).never()).receivePacket(any(Packet.class));
        }
    }

    @Test
    @DisplayName("Test handleClient xử lý kết nối trống")
    void testHandleClient_EmptyConnection() throws Exception {
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            int port = serverSocket.getLocalPort();
            
            Thread serverThread = new Thread(() -> {
                try {
                    Socket clientSocket = serverSocket.accept();
                    nodeGateway.startListening(testNodeInfo, port);
                    nodeGateway.handleClient(clientSocket);
                } catch (Exception e) {
                    // Expected
                }
            });
            serverThread.start();

            try (Socket clientSocket = new Socket("localhost", port)) {
                // Connect but send nothing - just close
            }

            serverThread.join(2000);

            // Verify no packet was received
            verify(mockTcpService, after(500).never()).receivePacket(any(Packet.class));
        }
    }

}
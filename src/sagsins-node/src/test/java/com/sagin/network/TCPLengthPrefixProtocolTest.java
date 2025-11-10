package com.sagin.network;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sagin.model.*;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.implement.TCP_Service;
import com.sagin.network.interfaces.ITCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.routing.IRoutingService;
import com.sagin.routing.RLRoutingService;
import com.sagin.service.BatchPacketService;
import com.sagin.service.INodeService;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Unit test to validate TCP length-prefix protocol implementation.
 * This test ensures that packets are correctly serialized with a 4-byte length prefix
 * and can be successfully transmitted and received.
 */
@ExtendWith(MockitoExtension.class)
class TCPLengthPrefixProtocolTest {

    @Mock
    private INodeRepository nodeRepository;
    
    @Mock
    private IUserRepository userRepository;
    
    @Mock
    private INodeService nodeService;
    
    @Mock
    private IRoutingService routingService;
    
    @Mock
    private BatchPacketService batchPacketService;
    
    @Mock
    private RLRoutingService rlRoutingService;
    
    private TCP_Service tcpService;
    private NodeGateway nodeGateway;
    private ServerSocket testServerSocket;
    private int testPort;

    @BeforeEach
    void setUp() throws IOException {
        // Create TCP service
        tcpService = new TCP_Service(
            nodeRepository,
            nodeService,
            userRepository,
            routingService,
            batchPacketService,
            rlRoutingService
        );
        
        // Find an available port for testing
        testServerSocket = new ServerSocket(0);
        testPort = testServerSocket.getLocalPort();
    }

    @AfterEach
    void tearDown() throws IOException {
        if (testServerSocket != null && !testServerSocket.isClosed()) {
            testServerSocket.close();
        }
        if (tcpService != null) {
            tcpService.stop();
        }
    }

    @Test
    @DisplayName("Test int to 4-byte array conversion")
    void testIntToBytesConversion() {
        // Test the intToBytes conversion logic (using reflection or a copy of the method)
        // We'll validate the protocol by simulating the conversion
        
        // Test case 1: Small number
        int value1 = 1024;
        byte[] bytes1 = intToBytes(value1);
        assertEquals(4, bytes1.length, "Should produce 4 bytes");
        
        // Convert back to verify
        int reconstructed1 = bytesToInt(bytes1);
        assertEquals(value1, reconstructed1, "Round-trip conversion should preserve value");
        
        // Test case 2: Larger number
        int value2 = 65536;
        byte[] bytes2 = intToBytes(value2);
        assertEquals(4, bytes2.length, "Should produce 4 bytes");
        
        int reconstructed2 = bytesToInt(bytes2);
        assertEquals(value2, reconstructed2, "Round-trip conversion should preserve value");
        
        // Test case 3: Maximum typical packet size
        int value3 = 16384; // 16 KB
        byte[] bytes3 = intToBytes(value3);
        assertEquals(4, bytes3.length, "Should produce 4 bytes");
        
        int reconstructed3 = bytesToInt(bytes3);
        assertEquals(value3, reconstructed3, "Round-trip conversion should preserve value");
    }

    @Test
    @DisplayName("Test length-prefix protocol with real packet serialization")
    void testLengthPrefixProtocol() throws Exception {
        // Create a test packet
        Packet packet = createTestPacket("TEST-001", "USER-A", "USER-B", "NODE-A", "NODE-B");
        
        // Serialize the packet
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        byte[] packetData = mapper.writeValueAsBytes(packet);
        
        // Create length prefix
        byte[] lengthPrefix = intToBytes(packetData.length);
        
        // Combine length prefix + packet data
        byte[] fullMessage = new byte[lengthPrefix.length + packetData.length];
        System.arraycopy(lengthPrefix, 0, fullMessage, 0, lengthPrefix.length);
        System.arraycopy(packetData, 0, fullMessage, lengthPrefix.length, packetData.length);
        
        // Simulate reading the message as the receiver would
        try (ByteArrayInputStream bais = new ByteArrayInputStream(fullMessage);
             DataInputStream dis = new DataInputStream(bais)) {
            
            // Read the length prefix
            int receivedLength = dis.readInt();
            assertEquals(packetData.length, receivedLength, "Length prefix should match actual data length");
            
            // Read the packet data
            byte[] receivedData = new byte[receivedLength];
            dis.readFully(receivedData);
            
            // Deserialize and verify
            Packet receivedPacket = mapper.readValue(receivedData, Packet.class);
            assertNotNull(receivedPacket, "Packet should be successfully deserialized");
            assertEquals(packet.getPacketId(), receivedPacket.getPacketId(), "Packet ID should match");
            assertEquals(packet.getSourceUserId(), receivedPacket.getSourceUserId(), "Source user should match");
            assertEquals(packet.getDestinationUserId(), receivedPacket.getDestinationUserId(), "Destination user should match");
        }
    }

    @Test
    @DisplayName("Test end-to-end packet transmission with actual TCP socket")
    void testEndToEndPacketTransmission() throws Exception {
        // Close the test server socket so the gateway can bind to the port
        testServerSocket.close();
        
        // Create a test node that will receive the packet
        NodeInfo testNode = createTestNode("TEST-NODE", "127.0.0.1", testPort);
        
        // Create a latch to wait for packet reception
        CountDownLatch receiveLatch = new CountDownLatch(1);
        AtomicReference<Packet> receivedPacketRef = new AtomicReference<>();
        
        // Create a custom TCP service that captures received packets
        ITCP_Service capturingService = new ITCP_Service() {
            @Override
            public void receivePacket(Packet packet) {
                receivedPacketRef.set(packet);
                receiveLatch.countDown();
            }
            
            @Override
            public void sendPacket(Packet packet, String senderNodeId) {
                // Not used in this test
            }
        };
        
        // Start a NodeGateway on the test port
        NodeGateway gateway = new NodeGateway(capturingService);
        
        // Start listening in a separate thread
        Thread listenerThread = new Thread(() -> {
            try {
                gateway.startListening(testNode, testPort);
            } catch (IOException e) {
                fail("Failed to start gateway: " + e.getMessage());
            }
        });
        listenerThread.start();
        
        // Give the server time to start
        Thread.sleep(200);
        
        // Create and send a test packet
        Packet testPacket = createTestPacket("E2E-001", "USER-A", "USER-B", "NODE-A", "TEST-NODE");
        testPacket.setNextHopNodeId("TEST-NODE");
        
        // Send the packet using TCP_Service's internal logic
        // We'll simulate this by manually creating a connection and sending
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        byte[] packetData = mapper.writeValueAsBytes(testPacket);
        
        try (Socket socket = new Socket("127.0.0.1", testPort)) {
            // Write length prefix + packet data (as TCP_Service does)
            byte[] lengthPrefix = intToBytes(packetData.length);
            socket.getOutputStream().write(lengthPrefix);
            socket.getOutputStream().write(packetData);
            socket.getOutputStream().flush();
        }
        
        // Wait for the packet to be received (with timeout)
        boolean received = receiveLatch.await(2, TimeUnit.SECONDS);
        assertTrue(received, "Packet should be received within timeout");
        
        // Verify the received packet
        Packet receivedPacket = receivedPacketRef.get();
        assertNotNull(receivedPacket, "Received packet should not be null");
        assertEquals(testPacket.getPacketId(), receivedPacket.getPacketId(), "Packet ID should match");
        assertEquals("TEST-NODE", receivedPacket.getCurrentHoldingNodeId(), "Current node should be set");
        
        // Clean up
        gateway.stopListening();
        listenerThread.interrupt();
    }

    // Helper methods

    /**
     * Converts an integer to a 4-byte array in big-endian format.
     * This mirrors the implementation in TCP_Service.
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
     * Converts a 4-byte array back to an integer (for verification).
     */
    private int bytesToInt(byte[] bytes) {
        return ((bytes[0] & 0xFF) << 24) |
               ((bytes[1] & 0xFF) << 16) |
               ((bytes[2] & 0xFF) << 8) |
               (bytes[3] & 0xFF);
    }

    private NodeInfo createTestNode(String nodeId, String ipAddress, int port) {
        NodeInfo node = new NodeInfo();
        node.setNodeId(nodeId);
        node.setNodeName(nodeId);
        node.setNodeType(NodeType.GROUND_STATION);
        node.setOperational(true);
        node.setHealthy(true);
        node.setBatteryChargePercent(80.0);
        node.setResourceUtilization(0.3);
        node.setPacketBufferCapacity(100);
        node.setCurrentPacketCount(10);
        node.setNodeProcessingDelayMs(2.0);
        node.setPacketLossRate(0.01);
        node.setLastUpdated(Instant.now());
        
        Position position = new Position();
        position.setLatitude(16.0);
        position.setLongitude(108.0);
        position.setAltitude(0.0);
        node.setPosition(position);
        
        Communication communication = new Communication();
        communication.setIpAddress(ipAddress);
        communication.setPort(port);
        communication.setBandwidthMHz(100.0);
        communication.setFrequencyGHz(2.4);
        communication.setProtocol("TCP");
        node.setCommunication(communication);
        
        return node;
    }

    private Packet createTestPacket(String packetId, String sourceUserId, String destUserId,
                                   String sourceStation, String destStation) {
        Packet packet = new Packet();
        packet.setPacketId(packetId);
        packet.setSourceUserId(sourceUserId);
        packet.setDestinationUserId(destUserId);
        packet.setStationSource(sourceStation);
        packet.setStationDest(destStation);
        packet.setTTL(15);
        packet.setPayloadSizeByte(1024);
        packet.setAccumulatedDelayMs(0.0);
        packet.setMaxAcceptableLatencyMs(1000.0);
        packet.setServiceQoS(new ServiceQoS(ServiceType.FILE_TRANSFER, 3, 1000.0, 50.0, 10.0, 0.05));
        packet.setUseRL(false);
        packet.setTimeSentFromSourceMs(System.currentTimeMillis());
        packet.setDropped(false);
        packet.setPathHistory(new ArrayList<>());
        return packet;
    }
}

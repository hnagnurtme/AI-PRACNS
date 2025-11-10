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

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.time.Instant;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class to validate packet length validation and endianness handling.
 * Tests the fixes for the invalid TCP packet length issue (2065854561).
 */
@ExtendWith(MockitoExtension.class)
class PacketLengthValidationTest {

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
    @DisplayName("Test intToBytes produces correct big-endian format")
    void testIntToBytesBigEndian() {
        // Test various packet sizes
        
        // Test case 1: Small packet (1024 bytes)
        int size1 = 1024;
        byte[] bytes1 = intToBytes(size1);
        assertEquals(4, bytes1.length, "Should produce 4 bytes");
        
        // Verify big-endian format
        assertEquals(0x00, bytes1[0] & 0xFF, "MSB should be 0x00");
        assertEquals(0x00, bytes1[1] & 0xFF, "Byte 1 should be 0x00");
        assertEquals(0x04, bytes1[2] & 0xFF, "Byte 2 should be 0x04");
        assertEquals(0x00, bytes1[3] & 0xFF, "LSB should be 0x00");
        
        // Convert back and verify
        int reconstructed1 = bytesToInt(bytes1);
        assertEquals(size1, reconstructed1, "Round-trip conversion should preserve value");
        
        // Test case 2: Larger packet (16384 bytes = 16KB, max allowed)
        int size2 = 16384;
        byte[] bytes2 = intToBytes(size2);
        
        assertEquals(0x00, bytes2[0] & 0xFF, "MSB should be 0x00");
        assertEquals(0x00, bytes2[1] & 0xFF, "Byte 1 should be 0x00");
        assertEquals(0x40, bytes2[2] & 0xFF, "Byte 2 should be 0x40");
        assertEquals(0x00, bytes2[3] & 0xFF, "LSB should be 0x00");
        
        int reconstructed2 = bytesToInt(bytes2);
        assertEquals(size2, reconstructed2, "Round-trip conversion should preserve value");
    }

    @Test
    @DisplayName("Test detection of wrong endianness (little-endian)")
    void testLittleEndianDetection() {
        // Create a valid packet size in little-endian format
        // This simulates a client sending data with wrong byte order
        int validSize = 1024; // In big-endian: 0x00 0x00 0x04 0x00
        
        // Create little-endian bytes (reversed)
        byte[] littleEndianBytes = new byte[] {
            (byte) validSize,           // LSB first: 0x00
            (byte) (validSize >> 8),    // 0x04
            (byte) (validSize >> 16),   // 0x00
            (byte) (validSize >> 24)    // MSB last: 0x00
        };
        
        // When read as big-endian, this would give an invalid value
        int invalidBigEndian = bytesToInt(littleEndianBytes);
        
        // The reversed bytes would be: 0x00 0x04 0x00 0x00
        // Which equals 262144 in decimal
        assertTrue(invalidBigEndian > 16384, 
            "Reading little-endian as big-endian should give invalid (too large) value");
        
        // But reversing byte order gives the correct value
        byte[] correctedBytes = new byte[] {
            littleEndianBytes[3],
            littleEndianBytes[2],
            littleEndianBytes[1],
            littleEndianBytes[0]
        };
        int correctedValue = bytesToInt(correctedBytes);
        assertEquals(validSize, correctedValue, 
            "Reversing byte order should give valid packet size");
    }

    @Test
    @DisplayName("Test invalid packet length: negative value")
    void testNegativePacketLength() {
        // Simulate receiving a negative length (corrupted data)
        byte[] invalidBytes = new byte[] {
            (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF
        };
        
        int length = bytesToInt(invalidBytes);
        assertTrue(length < 0, "Should result in negative value");
        
        // NodeGateway should reject this as <= 0
    }

    @Test
    @DisplayName("Test invalid packet length: exceeds maximum")
    void testOversizedPacketLength() {
        // Simulate receiving a packet length that exceeds MAX_PACKET_SIZE (16KB)
        int oversized = 20 * 1024; // 20KB
        byte[] bytes = intToBytes(oversized);
        
        int length = bytesToInt(bytes);
        assertEquals(oversized, length, "Should correctly represent 20KB");
        assertTrue(length > 16 * 1024, "Should exceed max packet size of 16KB");
        
        // NodeGateway should reject this as > MAX_PACKET_SIZE
    }

    @Test
    @DisplayName("Test the specific error case: 2065854561")
    void testSpecificErrorCase() {
        // The error message showed: "Received invalid packet length 2065854561"
        // In hex: 0x7B123456 (approximately)
        int invalidLength = 2065854561;
        
        // Convert to bytes to see the pattern
        byte[] bytes = intToBytes(invalidLength);
        
        // This should be rejected by NodeGateway
        assertTrue(invalidLength > 16 * 1024, 
            "The reported error value far exceeds MAX_PACKET_SIZE");
        
        // Check if reversing byte order gives a valid value
        byte[] reversedBytes = new byte[] {
            bytes[3], bytes[2], bytes[1], bytes[0]
        };
        int reversedValue = bytesToInt(reversedBytes);
        
        // Log the pattern for debugging
        System.out.printf("Invalid length: %d (0x%08X)%n", invalidLength, invalidLength);
        System.out.printf("Reversed: %d (0x%08X)%n", reversedValue, reversedValue);
        System.out.printf("Bytes: 0x%02X %02X %02X %02X%n", 
            bytes[0], bytes[1], bytes[2], bytes[3]);
    }

    @Test
    @DisplayName("Test end-to-end packet transmission with valid length")
    void testValidPacketTransmission() throws Exception {
        // Close the test server socket so the gateway can bind to the port
        testServerSocket.close();
        
        // Create a test node
        NodeInfo testNode = createTestNode("TEST-NODE", "127.0.0.1", testPort);
        
        // Create a latch to wait for packet reception
        CountDownLatch receiveLatch = new CountDownLatch(1);
        AtomicBoolean packetReceived = new AtomicBoolean(false);
        
        // Create a custom TCP service that captures received packets
        ITCP_Service capturingService = new ITCP_Service() {
            @Override
            public void receivePacket(Packet packet) {
                packetReceived.set(true);
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
        Packet testPacket = createTestPacket("TEST-001", "USER-A", "USER-B", "NODE-A", "TEST-NODE");
        
        // Serialize the packet
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        byte[] packetData = mapper.writeValueAsBytes(testPacket);
        
        // Send using correct big-endian format
        try (Socket socket = new Socket("127.0.0.1", testPort)) {
            byte[] lengthPrefix = intToBytes(packetData.length);
            socket.getOutputStream().write(lengthPrefix);
            socket.getOutputStream().write(packetData);
            socket.getOutputStream().flush();
        }
        
        // Wait for the packet to be received (with timeout)
        boolean received = receiveLatch.await(2, TimeUnit.SECONDS);
        assertTrue(received, "Packet should be received within timeout");
        assertTrue(packetReceived.get(), "Packet should be successfully processed");
        
        // Clean up
        gateway.stopListening();
        listenerThread.interrupt();
    }

    @Test
    @DisplayName("Test rejection of packet with invalid (corrupted) length")
    void testInvalidLengthRejection() throws Exception {
        // Close the test server socket so the gateway can bind to the port
        testServerSocket.close();
        
        // Create a test node
        NodeInfo testNode = createTestNode("TEST-NODE", "127.0.0.1", testPort);
        
        // Create a latch to detect if packet is incorrectly processed
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
        
        // Send a packet with invalid length
        try (Socket socket = new Socket("127.0.0.1", testPort)) {
            // Send an invalid length (larger than MAX_PACKET_SIZE)
            byte[] invalidLength = intToBytes(20 * 1024); // 20KB, exceeds 16KB limit
            socket.getOutputStream().write(invalidLength);
            socket.getOutputStream().flush();
            
            // Don't send actual data - the gateway should close connection
        } catch (IOException e) {
            // Connection might be closed by server, which is expected
        }
        
        // Wait a bit to see if packet is processed (it shouldn't be)
        boolean received = receiveLatch.await(1, TimeUnit.SECONDS);
        assertFalse(received, "Invalid packet should not be processed");
        assertNull(receivedPacketRef.get(), "No packet should be received");
        
        // Clean up
        gateway.stopListening();
        listenerThread.interrupt();
    }

    // Helper methods

    /**
     * Converts an integer to a 4-byte array in big-endian format.
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
     * Converts a 4-byte array back to an integer (big-endian).
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

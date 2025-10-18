package com.sagin.service.__test__;

import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.network.implement.TCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.routing.IRoutingService;
import com.sagin.service.INodeService;


import org.junit.jupiter.api.*;
import org.mockito.*;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Optional;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class TCP_ServiceTest {

    @Mock private INodeRepository nodeRepository;
    @Mock private IUserRepository userRepository;
    @Mock private INodeService nodeService;
    @Mock private IRoutingService routingService;

    private TCP_Service tcpService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        tcpService = new TCP_Service(nodeRepository, nodeService, userRepository, routingService);
    }

    @AfterEach
    void tearDown() {
        tcpService.stop();
    }

    // =====================================================
    // 🧩 TEST CASE 1: SEND SUCCESSFULLY
    // =====================================================
    @Test
    void testSendPacketSuccess() throws Exception {
        // Giả lập node kế tiếp có server socket thật
        int port = findFreePort();
        NodeInfo nextNode = new NodeInfo();
        nextNode.setHost("127.0.0.1");
        nextNode.setPort(port);

        Packet packet = new Packet();
        packet.setPacketId("P-001");
        packet.setNextHopNodeId("N-2");

        when(nodeRepository.getNodeInfo("N-2")).thenReturn(Optional.of(nextNode));

        // Tạo server giả (để nhận socket connect)
        Thread fakeServer = new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                Socket client = serverSocket.accept(); // chờ client connect
                client.getInputStream().readAllBytes(); // đọc hết
            } catch (IOException ignored) {}
        });
        fakeServer.start();

        // Gửi packet
        tcpService.sendPacket(packet, "N-1");

        // Chờ vài giây để scheduler chạy
        Thread.sleep(1500);

        // Kiểm tra nodeService.processSuccessfulSend() được gọi
        verify(nodeService, atLeastOnce()).processSuccessfulSend(eq("N-1"), any(Packet.class));
    }

    // =====================================================
    // 🧩 TEST CASE 2: NO ROUTE FOUND
    // =====================================================
    @Test
    void testReceivePacketNoRoute() {
        Packet packet = new Packet();
        packet.setPacketId("P-002");
        packet.setCurrentHoldingNodeId("N-A");
        packet.setStationDest("N-Z");

        when(routingService.getBestRoute("N-A", "N-Z")).thenReturn(null);

        tcpService.receivePacket(packet);

        assertTrue(packet.isDropped());
        assertEquals("NO_ROUTE_TO_HOST", packet.getDropReason());
    }

    // =====================================================
    // 🧩 TEST CASE 3: RETRY MAX -> DROP
    // =====================================================
    @Test
    void testSendPacketMaxRetryDrop() throws Exception {
        // Giả lập node đích unreachable
        NodeInfo nextNode = new NodeInfo();
        nextNode.setHost("127.0.0.1");
        nextNode.setPort(9999); // không có server thật
        Packet packet = new Packet();
        packet.setPacketId("P-003");
        packet.setNextHopNodeId("N-B");

        when(nodeRepository.getNodeInfo("N-B")).thenReturn(Optional.of(nextNode));

        tcpService.sendPacket(packet, "N-A");

        // Chờ đủ thời gian cho 5 lần retry (0.5s mỗi lần)
        Thread.sleep(3500);

        assertTrue(packet.isDropped());
        assertEquals("TCP_SEND_FAILED_MAX_RETRIES", packet.getDropReason());
    }

    // =====================================================
    // 🔧 Utility: tìm port trống
    // =====================================================
    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }
}

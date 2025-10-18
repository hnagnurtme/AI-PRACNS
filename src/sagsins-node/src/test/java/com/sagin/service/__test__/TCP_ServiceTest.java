package com.sagin.service.__test__;

import com.sagin.model.*;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.service.INodeService;
import com.sagin.routing.IRoutingService;
import com.sagin.network.implement.TCP_Service;

import org.junit.jupiter.api.*;
import org.mockito.*;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class TCP_ServiceTest {

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

    @Test
    void testSendPacketSuccess() throws Exception {
        int port = findFreePort();
        NodeInfo nextNode = new NodeInfo();
        nextNode.setHost("127.0.0.1");
        nextNode.setPort(port);

        Packet packet = new Packet();
        packet.setPacketId("P-001");
        packet.setNextHopNodeId("N-2");

        when(nodeRepository.getNodeInfo("N-2")).thenReturn(Optional.of(nextNode));

        CountDownLatch latch = new CountDownLatch(1);

        Thread fakeServer = new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                Socket client = serverSocket.accept();
                client.getInputStream().readAllBytes();
                latch.countDown(); // thông báo đã nhận
            } catch (IOException ignored) {}
        });
        fakeServer.start();

        tcpService.sendPacket(packet, "N-1");

        // chờ max 2s thay vì sleep
        assertTrue(latch.await(2, TimeUnit.SECONDS));

        verify(nodeService, atLeastOnce()).processSuccessfulSend(eq("N-1"), any(Packet.class));
        fakeServer.join(500); // đảm bảo thread kết thúc
    }

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

    @Test
    void testSendPacketMaxRetryDrop() throws Exception {
        NodeInfo nextNode = new NodeInfo();
        nextNode.setHost("127.0.0.1");
        nextNode.setPort(9999); // unreachable
        Packet packet = new Packet();
        packet.setPacketId("P-003");
        packet.setNextHopNodeId("N-B");

        when(nodeRepository.getNodeInfo("N-B")).thenReturn(Optional.of(nextNode));

        tcpService.sendPacket(packet, "N-A");

        // Chờ tối đa 4s, TCP_Service sẽ retry
        Thread.sleep(4000);

        assertTrue(packet.isDropped());
        assertEquals("TCP_SEND_FAILED_MAX_RETRIES", packet.getDropReason());
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }
}

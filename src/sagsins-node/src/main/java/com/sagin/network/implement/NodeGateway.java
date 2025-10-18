package com.sagin.network.implement;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.core.JsonProcessingException; 
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.network.interfaces.INodeGateway;
import com.sagin.network.interfaces.ITCP_Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException; 
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean; 

/**
 * Triển khai INodeGateway.
 * Mở một cổng TCP Server để lắng nghe các Packet đến node này.
 */
public class NodeGateway implements INodeGateway {

    private static final Logger logger = LoggerFactory.getLogger(NodeGateway.class);
    private final ITCP_Service tcpService;
    private final ObjectMapper objectMapper;
    private ServerSocket serverSocket;
    private Thread listenerThread;
    private ExecutorService clientHandlerPool; 
    private final AtomicBoolean isRunning = new AtomicBoolean(false); 
    private String nodeId; 
    public NodeGateway(ITCP_Service tcpService) {
        this.tcpService = tcpService;
        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
    }

    /**
     * Bắt đầu lắng nghe trên một cổng TCP cụ thể.
     * Mở ServerSocket và khởi động các luồng xử lý.
     *
     * @param info Thông tin của node hiện tại (chủ yếu để lấy nodeId).
     * @param port Cổng để lắng nghe.
     */
    @Override
    public void startListening(NodeInfo info, int port) {
        if (info == null || info.getNodeId() == null) {
            logger.error("[NodeGateway] Không thể bắt đầu: NodeInfo hoặc NodeId bị null.");
            return;
        }
        this.nodeId = info.getNodeId();

        // Chỉ bắt đầu nếu chưa chạy
        if (isRunning.compareAndSet(false, true)) {
            try {
                serverSocket = new ServerSocket(port);
                // Tạo một thread pool với số luồng động (cached) để xử lý client
                clientHandlerPool = Executors.newCachedThreadPool();
                logger.info("[NodeGateway] Node {} đang lắng nghe trên cổng {}...", nodeId, port);

                // Tạo và khởi động luồng lắng nghe chính
                listenerThread = new Thread(this::runListenerLoop, "NodeGateway-Listener-" + nodeId);
                listenerThread.start();

            } catch (IOException e) {
                logger.error("[NodeGateway] Node {}: Không thể mở cổng {}: {}", nodeId, port, e.getMessage());
                isRunning.set(false); // Đặt lại cờ nếu không khởi động được
            }
        } else {
            logger.warn("[NodeGateway] Node {} đã đang chạy trên cổng {}.", nodeId, serverSocket.getLocalPort());
        }
    }

    /**
     * Vòng lặp chính của luồng lắng nghe.
     * Chấp nhận kết nối mới và giao cho client handler.
     */
    private void runListenerLoop() {
        while (isRunning.get()) {
            try {
                // Chấp nhận kết nối mới (blocking)
                Socket clientSocket = serverSocket.accept();
                logger.debug("[NodeGateway] Node {}: Đã chấp nhận kết nối từ {}", nodeId, clientSocket.getRemoteSocketAddress());
                
                // Giao việc xử lý client này cho một thread trong pool
                clientHandlerPool.submit(() -> handleClient(clientSocket));
                
            } catch (SocketException se) {
                // Thường xảy ra khi serverSocket.close() được gọi từ stopListening()
                if (isRunning.get()) { // Chỉ log lỗi nếu không phải do cố ý dừng
                    logger.error("[NodeGateway] Node {}: Lỗi Socket khi chấp nhận kết nối: {}", nodeId, se.getMessage());
                } else {
                    logger.info("[NodeGateway] Node {}: Đã dừng chấp nhận kết nối.", nodeId);
                }
            } catch (IOException e) {
                if (isRunning.get()){
                    logger.error("[NodeGateway] Node {}: Lỗi I/O khi chấp nhận kết nối: {}", nodeId, e.getMessage());
                }
            }
        }
        logger.info("[NodeGateway] Node {}: Luồng lắng nghe đã kết thúc.", nodeId);
    }

    /**
     * Xử lý dữ liệu từ một client socket cụ thể.
     * Đọc dữ liệu, deserialize thành Packet, và gọi tcpService.receivePacket().
     *
     * @param clientSocket Socket của client vừa kết nối.
     */
    public void handleClient(Socket clientSocket) {
        // Sử dụng try-with-resources để đảm bảo stream và socket được đóng
        try (InputStream inputStream = clientSocket.getInputStream()) {
            byte[] buffer = new byte[4096];
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                baos.write(buffer, 0, read);
            }
            byte[] data = baos.toByteArray();

            if (data.length == 0) {
                logger.warn("[NodeGateway] Node {}: Nhận được kết nối trống từ {}. Đóng.", nodeId, clientSocket.getRemoteSocketAddress());
                return;
            }

            logger.debug("[NodeGateway] Node {}: Đã nhận {} bytes từ {}. Đang deserialize...", nodeId, data.length, clientSocket.getRemoteSocketAddress());

            // Deserialize byte array thành đối tượng Packet
            Packet receivedPacket = objectMapper.readValue(data, Packet.class);

            // **QUAN TRỌNG:** Ghi đè/Cập nhật node đang giữ packet này
            receivedPacket.setCurrentHoldingNodeId(this.nodeId);

            // Giao packet cho TCP_Service xử lý
            tcpService.receivePacket(receivedPacket);

        } catch (JsonProcessingException jpe) {
            logger.error("[NodeGateway] Node {}: Lỗi deserialize JSON từ {}: {}", nodeId, clientSocket.getRemoteSocketAddress(), jpe.getMessage());
        } catch (IOException e) {
            logger.error("[NodeGateway] Node {}: Lỗi I/O khi đọc từ {}: {}", nodeId, clientSocket.getRemoteSocketAddress(), e.getMessage());
        } finally {
            try {
                clientSocket.close(); // Đảm bảo đóng socket client
            } catch (IOException e) {
                logger.warn("[NodeGateway] Node {}: Lỗi khi đóng client socket: {}", nodeId, e.getMessage());
            }
        }
    }


    /**
     * Dừng lắng nghe và giải phóng tài nguyên.
     * Đóng ServerSocket và shutdown các luồng.
     */
    @Override
    public void stopListening() {
        if (isRunning.compareAndSet(true, false)) {
            logger.info("[NodeGateway] Node {}: Đang dừng lắng nghe...", nodeId);
            try {
                if (serverSocket != null && !serverSocket.isClosed()) {
                    serverSocket.close(); // Gây SocketException để ngắt accept()
                }
            } catch (IOException e) {
                logger.error("[NodeGateway] Node {}: Lỗi khi đóng ServerSocket: {}", nodeId, e.getMessage());
            }

            // Dừng thread pool xử lý client
            if (clientHandlerPool != null) {
                clientHandlerPool.shutdown(); // Ngừng nhận task mới
                try {
                    // Chờ tối đa 5 giây để các task đang chạy hoàn thành
                    if (!clientHandlerPool.awaitTermination(5, TimeUnit.SECONDS)) {
                        clientHandlerPool.shutdownNow(); // Buộc dừng nếu quá lâu
                    }
                } catch (InterruptedException ie) {
                    clientHandlerPool.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }

            // Chờ luồng lắng nghe chính kết thúc
            if (listenerThread != null) {
                try {
                    listenerThread.join(1000); // Chờ tối đa 1 giây
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            logger.info("[NodeGateway] Node {}: Đã dừng hoàn toàn.", nodeId);
        } else {
            logger.warn("[NodeGateway] Node {}: Gateway không đang chạy.", nodeId);
        }
    }
}
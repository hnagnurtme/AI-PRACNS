package com.sagin.core.service;

import com.sagin.core.INodeGatewayService;
import com.sagin.core.INodeService;
import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.PacketTransferWrapper;
import com.sagin.util.PacketSerializerHelper;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Triển khai lắng nghe TCP thực sự.
 * Lắng nghe các kết nối client và đọc đối tượng Packet được gửi đến.
 */
public class TcpGatewayService implements INodeGatewayService, Runnable {

    private static final Logger logger = LoggerFactory.getLogger(TcpGatewayService.class);

    // Luồng dịch vụ để xử lý nhiều kết nối client cùng lúc
    private final ExecutorService clientHandlerPool = Executors.newCachedThreadPool();

    private INodeService nodeServiceReference;
    private NodeInfo selfNodeInfo;
    private ServerSocket serverSocket;
    private volatile boolean isRunning = false;
    private int listeningPort = -1;

    /** @inheritdoc */
    @Override
    public void setNodeServiceReference(INodeService service) {
        this.nodeServiceReference = service;
    }

    /** @inheritdoc */
    @Override
    public void startListening(NodeInfo info, int port) {
        if (isRunning)
            return;

        this.selfNodeInfo = info;
        this.listeningPort = port;

        try {
            this.serverSocket = new ServerSocket(listeningPort);
            this.isRunning = true;

            // Khởi chạy luồng chính để chấp nhận kết nối (Acceptor Thread)
            new Thread(this, "TcpAcceptor-" + info.getNodeId()).start();
            logger.info("[Gateway {}] Server Listener đã khởi động thành công trên cổng {}.", info.getNodeId(), port);

        } catch (IOException e) {
            logger.error("[Gateway {}] LỖI không thể khởi động ServerSocket trên cổng {}: {}", info.getNodeId(), port,
                    e.getMessage());
            this.isRunning = false;
        }
    }

    /**
     * Luồng chính (Acceptor Thread) để lắng nghe các kết nối đến.
     */
    @Override
    public void run() {
        while (isRunning) {
            try {
                // 1. Chấp nhận kết nối client mới
                Socket clientSocket = serverSocket.accept();
                logger.info("[Gateway {}] Đã chấp nhận kết nối từ Client: {}", selfNodeInfo.getNodeId(),
                        clientSocket.getInetAddress().getHostAddress());

                // 2. Chuyển giao việc xử lý cho Thread Pool
                clientHandlerPool.submit(new ClientHandler(clientSocket));

            } catch (IOException e) {
                if (isRunning) {
                    logger.error("[Gateway {}] LỖI chấp nhận kết nối: {}", selfNodeInfo.getNodeId(), e.getMessage());
                } else {
                    // Lỗi xảy ra khi socket bị đóng (stopListening được gọi)
                    logger.info("[Gateway {}] ServerSocket đã đóng.", selfNodeInfo.getNodeId());
                }
            }
        }
    }

    /** @inheritdoc */
    @Override
    public void stopListening() {
        this.isRunning = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
            clientHandlerPool.shutdown();
            if (!clientHandlerPool.awaitTermination(5, TimeUnit.SECONDS)) {
                clientHandlerPool.shutdownNow();
            }
            logger.info("[Gateway {}] Server Listener đã dừng hoạt động.", selfNodeInfo.getNodeId());
        } catch (IOException | InterruptedException e) {
            logger.error("[Gateway {}] LỖI khi dừng ServerSocket: {}", selfNodeInfo.getNodeId(), e.getMessage());
        }
    }

    private class ClientHandler implements Runnable {
        private final Socket clientSocket;

        public ClientHandler(Socket socket) {
            this.clientSocket = socket;
        }

        @Override
        public void run() {
            String clientAddress = clientSocket.getInetAddress().getHostAddress();
            logger.info("[Gateway {}] Bắt đầu xử lý kết nối từ Client: {}", selfNodeInfo.getNodeId(), clientAddress);

            // THAY THẾ ObjectInputStream bằng BufferedReader (để đọc JSON String)
            try (
                    InputStreamReader isr = new InputStreamReader(clientSocket.getInputStream(), "UTF-8"); // Đảm bảo mã
                                                                                                           // hóa
                    BufferedReader reader = new BufferedReader(isr)) {

                // --- 1. ĐỌC TOÀN BỘ CHUỖI JSON TỪ LUỒNG ---
                StringBuilder jsonBuilder = new StringBuilder();
                String line;

                // Đọc từng dòng cho đến khi luồng kết thúc (hoặc kết nối bị đóng)
                while ((line = reader.readLine()) != null) {
                    jsonBuilder.append(line);
                }

                String jsonString = jsonBuilder.toString().trim(); // Lấy JSON và xóa khoảng trắng thừa

                logger.info("Raw JSON input: {}", jsonString);

                if (jsonString.isEmpty()) {
                    logger.warn("[Gateway {}] Client {} đóng kết nối mà không gửi dữ liệu JSON.",
                            selfNodeInfo.getNodeId(), clientAddress);
                    return;
                }

                // 2. Giải tuần tự hóa JSON thành đối tượng WRAPPER
                // Sử dụng PacketTransferWrapper.class vì nó là payload mới qua Socket
                PacketTransferWrapper wrapper = PacketSerializerHelper.deserialize(jsonString,
                        PacketTransferWrapper.class);

                // 3. Kiểm tra tính hợp lệ của Wrapper và Packet bên trong
                if (wrapper != null && wrapper.getPacket() != null) {

                    Packet packet = wrapper.getPacket();
                    LinkMetric linkMetric = wrapper.getLinkMetric();
                    if (linkMetric != null) {
                        linkMetric.calculateLinkScore();
                    }
                    logger.info("LinekedMetric after score calculation: {}", (linkMetric != null ? linkMetric.toString() : "null"));
                    // Log
                    logger.info("[Gateway {}] Nhận được JSON hợp lệ từ Client {}: Gói {}, Loại {}, LinkMetric: {}",
                            selfNodeInfo.getNodeId(), clientAddress, packet.getPacketId(), packet.getType(),
                            (linkMetric != null ? linkMetric.toString() : "null"));

                    // 4. Cập nhật thông tin node hiện tại
                    packet.setCurrentHoldingNodeId(selfNodeInfo.getNodeId());
                    // Cập nhật thời gian nhận gói tin tại Gateway
                    packet.setTimeSentFromSourceMs(System.currentTimeMillis());

                    logger.info("[Gateway {}] Nhận được gói DATA {} (Type: {}) từ Client.",
                            selfNodeInfo.getNodeId(), packet.getPacketId(), packet.getType());

                    // 5. Đưa gói tin vào luồng xử lý của Node Service
                    if (nodeServiceReference != null) {
                        logger.info("[Gateway {}] Chuyển gói {} đến NodeService để xử lý.",
                                selfNodeInfo.getNodeId(), packet.getPacketId());
                        nodeServiceReference.receivePacket(packet, linkMetric);
                    }

                } else {
                    // Lỗi deserialize đã được log chi tiết trong PacketSerializerHelper
                    logger.warn(
                            "[Gateway {}] KHÔNG THỂ PHÂN TÍCH: Client {} đã gửi JSON không hợp lệ. Chuỗi nhận được: [{}]",
                            selfNodeInfo.getNodeId(), clientAddress, jsonString);
                }

            } catch (IOException e) {
                // Lỗi này xảy ra khi kết nối bị ngắt đột ngột
                logger.error("[Gateway {}] LỖI I/O khi đọc dữ liệu từ client {}: {}",
                        selfNodeInfo.getNodeId(), clientAddress, e.getMessage());
            } finally {
                try {
                    // Đóng socket Client
                    clientSocket.close();
                } catch (IOException e) {
                    logger.error("Lỗi đóng socket client: {}", e.getMessage());
                }
            }
        }
    }
}
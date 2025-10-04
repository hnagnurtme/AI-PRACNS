package com.sagin.core.service;

import com.sagin.core.INodeGatewayService;
import com.sagin.core.INodeService;
import com.sagin.core.IPacketService;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.util.ProjectConstant;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Triển khai Gateway sử dụng TCP ServerSocket để lắng nghe kết nối từ Client ngoài.
 * Mỗi kết nối Client sẽ được xử lý trong một luồng riêng.
 */
public class TcpNodeGateway implements INodeGatewayService {

    private static final Logger logger = LoggerFactory.getLogger(TcpNodeGateway.class);
    
    private final IPacketService packetService; 
    private INodeService nodeServiceReference; 
    private ServerSocket serverSocket;
    private ExecutorService threadPool;
    private volatile boolean isRunning = false;

    public TcpNodeGateway( IPacketService packetService) {
        this.packetService = packetService;
        this.threadPool = Executors.newCachedThreadPool(); 
    }

    @Override
    public void startListening(NodeInfo info, int port) {
        if (isRunning) {
            logger.warn("Gateway {} đã chạy trên cổng {}.", info.getNodeId(), port);
            return;
        }

        isRunning = true;
        
        threadPool.submit(() -> {
            try {
                serverSocket = new ServerSocket(port);
                logger.info("GATEWAY {}: Bắt đầu lắng nghe TCP trên cổng {}.", info.getNodeId(), port);

                while (isRunning) {
                    Socket clientSocket = serverSocket.accept(); 
                    threadPool.submit(() -> handleClientConnection(clientSocket, info.getNodeId())); 
                }
            } catch (IOException e) {
                if (isRunning) {
                    logger.error("LỖI LẮNG NGHE TCP trên cổng {}: {}", port, e.getMessage());
                }
            }
        });
    }

    private void handleClientConnection(Socket clientSocket, String gatewayNodeId) {
        logger.info("TCP: Client mới kết nối từ {}", clientSocket.getInetAddress().getHostAddress());
        
        try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {
            String clientData;
            
            // Đọc dữ liệu từ Client (mô phỏng luồng TCP)
            while ((clientData = in.readLine()) != null) { 
                
                // --- PHẦN XỬ LÝ DỮ LIỆU CỐT LÕI (Mô hình hóa TCP -> Packet) ---
                
                // Giả định dữ liệu nhận được là một chuỗi JSON (hoặc chỉ là một dấu hiệu kích hoạt)
                // Vì không có logic parsing JSON phức tạp, ta chỉ mô phỏng việc tạo gói tin.
                
                Packet newPacket = packetService.generatePacket(
                    "CLIENT_EXTERNAL", // ID Client nguồn
                    "USER_TERMINAL_02", // Đích cuối (giả định)
                    ProjectConstant.SERVICE_TYPE_DATA_BULK // Gói tin dữ liệu thô
                );
                
                // Đặt CurrentHoldingNodeId là Node Gateway này
                newPacket.setCurrentHoldingNodeId(gatewayNodeId);
                
                if(nodeServiceReference == null){
                    logger.error("LỖI: NodeService tham chiếu chưa được thiết lập trong Gateway {}.", gatewayNodeId);
                    continue; // Bỏ qua gói tin nếu không có NodeService
                }
                // Đưa gói tin vào NodeService để bắt đầu luồng định tuyến
                nodeServiceReference.receivePacket(newPacket);
                
                logger.debug("TCP: Nhận dữ liệu. Đã tạo và đưa gói {} vào buffer.", newPacket.getPacketId());
            }
        } catch (IOException e) {
            logger.error("Lỗi xử lý kết nối Client: {}", e.getMessage());
        } finally {
            try {
                clientSocket.close();
            } catch (IOException e) { /* Bỏ qua lỗi đóng socket */ }
        }
    }

    @Override
    public void stopListening() {
        isRunning = false;
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
                logger.error("Lỗi đóng ServerSocket:", e);
            }
        }
        if (threadPool != null) {
            threadPool.shutdownNow();
        }
    }

    @Override
    public void setNodeServiceReference(INodeService service){
        this.nodeServiceReference = service; 
        logger.info("TCP Gateway đã nhận được tham chiếu NodeService.");

    }
}
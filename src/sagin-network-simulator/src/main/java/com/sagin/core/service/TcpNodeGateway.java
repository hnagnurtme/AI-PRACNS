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
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
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
        
        // Vòng lặp này chỉ chạy MỘT LẦN (cho một dòng dữ liệu từ client)
        while ((clientData = in.readLine()) != null) { 
            
            // --- PHẦN XỬ LÝ DỮ LIỆU CỐT LÕI ---
            
            // Hiện tại, ta vẫn dùng logic hardcode để tạo gói tin
            String payloadBase64 = "SGVsbG8gV29ybGQ=";
            List<String> immutableHistory = Arrays.asList("R1", "S2");
            // BƯỚC KHẮC PHỤC: Tạo một ArrayList mới từ danh sách cố định.
            List<String> mutableHistory = new ArrayList<>(immutableHistory); 

            Packet newPacket = new Packet(
                "PKT-XYZ-005",             // packetId
                "client-101",              // sourceUserId
                "UAV_001",                // destinationUserId
                System.currentTimeMillis(),// timestamp
                payloadBase64,             // payloadDataBase64
                11,                        // payloadSizeByte
                "API_REQUEST",             // serviceType
                15,                        // TTL
                "Router-C",                // currentHoldingNodeId
                "UAV_001",                // nextHopNodeId
                mutableHistory,                   // pathHistory
                0.0,                       // accumulatedDelayMs
                8,                         // priorityLevel
                500.0,                     // maxAcceptableLatencyMs
                0.01,                      // maxAcceptableLossRate
                false                      // dropped
            );
            
            logger.info("TCP: Tạo gói tin mới từ dữ liệu Client: {}", newPacket);
            
            if(nodeServiceReference == null){
                logger.error("LỖI: NodeService tham chiếu chưa được thiết lập trong Gateway {}.", gatewayNodeId);
                // Nếu lỗi, nên ngắt luôn thay vì tiếp tục
                break; 
            }
            
            // // Đưa gói tin vào NodeService để bắt đầu luồng định tuyến
            // nodeServiceReference.receivePacket(newPacket);
            // Trong handleClientConnection
            try {
                nodeServiceReference.receivePacket(newPacket); // <--- Bọc lệnh gọi này
            } catch (Exception e) {
                logger.error("LỖI KHÔNG MONG MUỐN khi gọi receivePacket: {}", e.getMessage(), e);
                // Gói tin bị drop nếu NodeService bị lỗi khi nhận
                newPacket.markDropped(); 
            }
            
            logger.info("TCP: Nhận dữ liệu. Đã tạo và đưa gói {} vào buffer.", newPacket.getPacketId());
            
            // NGẮT VÒNG LẶP: Đảm bảo chỉ xử lý 1 gói tin rồi thoát.
            break; 
        } 
        // Sau khi break, code sẽ thoát khỏi try block.
        
    } catch (IOException e) {
        logger.error("Lỗi xử lý kết nối Client: {}", e.getMessage());
    } finally {
        // Luôn đảm bảo đóng socket sau khi xử lý xong (hoặc sau khi có lỗi)
        try {
            clientSocket.close();
            logger.info("TCP: Đã đóng kết nối Client từ {}", clientSocket.getInetAddress().getHostAddress());
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
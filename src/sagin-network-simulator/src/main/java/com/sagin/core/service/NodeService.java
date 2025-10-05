package com.sagin.core.service;

import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.RoutingTable;
import com.sagin.core.ILinkManagerService;
import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeGatewayService;
import com.sagin.core.INodeService;
import com.sagin.routing.RoutingEngine;
import com.sagin.util.HostPort;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

// SLF4J Imports
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class NodeService implements INodeService { 

    private static final Logger logger = LoggerFactory.getLogger(NodeService.class);

    private final NodeInfo currentNodeInfo;
    private final Queue<Packet> packetBuffer;
    
    private RoutingTable routingTable; 
    
    // Services
    private final RoutingEngine routingEngine;
    private final ILinkManagerService linkManager; 
    private final INetworkManagerService networkManager; 
    private final INodeGatewayService gatewayService ;
    
    private final Map<String, NodeInfo> neighborNodes; 
    private final Map<String, com.sagin.model.LinkMetric> neighborLinkMetrics;

    
    // THÀNH PHẦN ĐA LUỒNG: Scheduler cho các tác vụ định kỳ của Node
    private final ScheduledExecutorService scheduler;
    
    // Có thể dùng một Executor khác để mô phỏng CPU xử lý gói tin tốc độ cao hơn
    // private final ExecutorService packetProcessor; 


    public NodeService(NodeInfo initialInfo, 
                    INetworkManagerService networkManager, 
                    RoutingEngine routingEngine, 
                    ILinkManagerService linkManager,
                    INodeGatewayService gatewayService) { 
        this.currentNodeInfo = initialInfo;
        this.packetBuffer = new ConcurrentLinkedQueue<>();
        this.routingTable = new RoutingTable(); 

        this.networkManager = networkManager;
        this.routingEngine = routingEngine;
        this.linkManager = linkManager;
        this.gatewayService = gatewayService;
        
        this.neighborNodes = new ConcurrentHashMap<>();
        this.neighborLinkMetrics = new ConcurrentHashMap<>();
        
        this.scheduler = Executors.newSingleThreadScheduledExecutor();

        logger.info("Service cho Node {} đã khởi tạo.", initialInfo.getNodeId());
    }

    @Override
    public void startSimulationLoop() {
        logger.info("Bắt đầu vòng lặp mô phỏng Node {}...", currentNodeInfo.getNodeId());
        
        discoverNeighborsAndRunRouting(); 

        final int EXTERNAL_PORT = HostPort.port; 
            
        logger.info("Khởi động Gateway trên cổng {}...", 
                        currentNodeInfo.getNodeId(), EXTERNAL_PORT);
        gatewayService.startListening(currentNodeInfo, EXTERNAL_PORT);
        
        // Lập lịch cho tác vụ chính (bao gồm cập nhật trạng thái và xử lý buffer)
        scheduler.scheduleAtFixedRate(() -> {
            try {
                if (currentNodeInfo.isOperational()) {
                    updateNodeState();
                    processBuffer(); 
                } else {
                    scheduler.shutdown();
                    logger.warn("Node {} bị shutdown.", currentNodeInfo.getNodeId());
                }
            } catch (Exception e) {
                logger.error("Lỗi nghiêm trọng trong vòng lặp mô phỏng của Node {}: {}", 
                            currentNodeInfo.getNodeId(), e.getMessage(), e);
            }
        }, 0, 1000, TimeUnit.MILLISECONDS); 
    }

    @Override
    public void receivePacket(Packet packet) {
        if (currentNodeInfo.isHealthy()) {
            packet.addToPath(currentNodeInfo.getNodeId());
            packet.setCurrentHoldingNodeId(currentNodeInfo.getNodeId());

            packetBuffer.offer(packet);
            currentNodeInfo.setPacketBufferLoad(packetBuffer.size());
            
            logger.info("{} nhận gói {} (Buffer: {})", 
                        currentNodeInfo.getNodeId(), 
                        packet.getPacketId(), 
                        packetBuffer.size());
        } else {
            packet.markDropped();
            logger.warn("WARNING: {} DROP gói {} (Quá tải/Hỏng hóc)", 
                        currentNodeInfo.getNodeId(), 
                        packet.getPacketId());
        }
    }
    
    private void processBuffer() {
        Packet packetToSend = packetBuffer.poll();
        if (packetToSend == null) {
            currentNodeInfo.setPacketBufferLoad(0);
            return;
        }

        // 1. Kiểm tra đích cuối
        if (packetToSend.getDestinationUserId().equals("UserID_Target")) { 
            logger.info("SUCCESS: {} Gói {} đã đến đích cuối.", 
                        currentNodeInfo.getNodeId(), 
                        packetToSend.getPacketId());
            return;
        }

        // 2. Quyết định định tuyến
        String nextHop = decideNextHop(packetToSend);
        if( nextHop == null ) {
            // logger.warn("WARNING: {} không tìm thấy Next Hop cho gói {}. Gói bị drop.", 
            //             currentNodeInfo.getNodeId(), 
            //             packetToSend.getPacketId());
            logger.warn("WARNING: {} không tìm thấy Next Hop cho gói {}.", 
                        currentNodeInfo.getNodeId(), 
                        packetToSend.getPacketId());
            nextHop = "localhost:3001"; // Mặc định gửi về một địa chỉ an toàn"
            logger.warn("Gửi gói {} tới địa chỉ an toàn mặc định {}", 
                        packetToSend.getPacketId(),
                        nextHop);
            // packetToSend.markDropped(); 
            // return;
        }
        
        if (nextHop != null) {
            sendPacket(packetToSend, nextHop);
        } else {
            logger.warn("WARNING: {} không tìm thấy Next Hop hợp lệ cho gói {}", 
                        currentNodeInfo.getNodeId(), 
                        packetToSend.getPacketId());
            packetToSend.markDropped(); 
        }
    }

    @Override
    public String decideNextHop(Packet packet) {
        // Đây là nơi logic lập lịch CPU sẽ ảnh hưởng đến độ trễ
        return routingEngine.getNextHop(packet, routingTable); 
    }

    @Override
    public void sendPacket(Packet packet, String nextHopId) {
        packet.decrementTTL();
        
        logger.info("{} đang gửi gói {} -> {}", 
                    currentNodeInfo.getNodeId(), 
                    packet.getPacketId(), 
                    nextHopId);
        // *Chú ý: Logic chuyển gói tin giữa các container/luồng sẽ do NetworkManager xử lý.*
        networkManager.transferPacket(packet, nextHopId);

    }

    @Override
    public void updateNodeState() {
        // Cập nhật LinkMetric động (mô phỏng nhiễu, chuyển động)
        neighborLinkMetrics.forEach((id, metric) -> 
            linkManager.updateDynamicMetrics(metric)
        );

        // Chạy lại thuật toán định tuyến và gán trực tiếp kết quả
        this.routingTable = routingEngine.computeRoutes(currentNodeInfo, neighborLinkMetrics); 

        currentNodeInfo.setLastUpdated(System.currentTimeMillis());
        currentNodeInfo.setPacketBufferLoad(packetBuffer.size());
    }

    private void discoverNeighborsAndRunRouting() {
        // ID của node láng giềng (ví dụ) mà Node này dự kiến kết nối
        final String NEIGHBOR_ID = "GS_001";
        
        // 1. TRUY VẤN MẠNG: Lấy thông tin láng giềng từ NetworkManager
        NodeInfo neighbor = networkManager.getNodeInfo(NEIGHBOR_ID);

        if (neighbor != null) {
            // Lưu thông tin NodeInfo của láng giềng
            neighborNodes.put(neighbor.getNodeId(), neighbor);
            
            // 2. TÍNH TOÁN LINK BAN ĐẦU
            LinkMetric initialLink = linkManager.calculateInitialMetric(
                currentNodeInfo.getPosition(), 
                neighbor.getPosition()
            );
            
            // Thiết lập ID và tính Score
            initialLink.setSourceNodeId(currentNodeInfo.getNodeId());
            initialLink.setDestinationNodeId(neighbor.getNodeId());
            initialLink.calculateLinkScore();
            
            // Lưu Link Metric vào Map của Node hiện tại
            neighborLinkMetrics.put(NEIGHBOR_ID, initialLink);

            logger.info("Node {} đã khám phá {} láng giềng, kết nối với {}.", 
                        currentNodeInfo.getNodeId(), 
                        neighborNodes.size(),
                        NEIGHBOR_ID);

            // 3. CHẠY ĐỊNH TUYẾN
            // Gán trực tiếp kết quả tính toán vào bảng định tuyến
            this.routingTable = routingEngine.computeRoutes(currentNodeInfo, neighborLinkMetrics); 
            
        } else {
            logger.warn("Node {} không tìm thấy Node láng giềng {} để khởi tạo định tuyến.", 
                        currentNodeInfo.getNodeId(), NEIGHBOR_ID);
            // Khởi tạo bảng định tuyến trống nếu không có láng giềng
            this.routingTable = new com.sagin.model.RoutingTable();
        }
    }
}
package com.sagin.util;

import com.sagin.configuration.FireStoreConfiguration;
import com.sagin.core.ILinkManagerService;
import com.sagin.core.INodeService;
import com.sagin.core.IUserService;
import com.sagin.core.INodeGatewayService;
import com.sagin.core.service.NodeService;
import com.sagin.core.service.TcpGatewayService; 
import com.sagin.core.service.UserService;
import com.sagin.core.service.LinkManagerService;
import com.sagin.core.service.NetworkManagerService; 
import com.sagin.model.*;
import com.sagin.repository.FirebaseNodeRepository;
import com.sagin.repository.INodeRepository;
import com.sagin.routing.DijkstraRoutingEngine;
import com.sagin.routing.IRoutingEngine;
import com.sagin.seeding.NodeSeeder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.UnknownHostException;


public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);
    private static final int FALLBACK_PORT = 8080; 

    public static void main(String[] args) {
        String nodeId;

        // --- LOGIC MẶC ĐỊNH VÀ XỬ LÝ THAM SỐ ---
        if (args.length < 1) { // Chỉ kiểm tra 1 tham số
            nodeId = "GS_LONDON";
            logger.warn("Thiếu tham số khởi chạy. Sử dụng mặc định: NODE_ID={}", nodeId);
        } else {
            nodeId = args[0];
        }
        // ----------------------------------------

        logger.info("--- KHỞI CHẠY NODE ĐỘC LẬP: {} ---", nodeId);
        new SimulationMain().runSingleNode(nodeId); // Chỉ truyền NODEID
    }
    
    public void runSingleNode(String nodeId) {

        try {
            FireStoreConfiguration.init(); 
            logger.info("Firebase SDK đã được khởi tạo thành công.");
        } catch (Exception e) {
            logger.error("LỖI KHỞI TẠO FIREBASE: Không thể tải cấu hình SDK.", e);
            return;
        }
        
        // --- 1. KHỞI TẠO REPOSITORY VÀ SEEDING DỮ LIỆU ---
        INodeRepository nodeRepository = new FirebaseNodeRepository();
        checkAndSeedDatabase(nodeRepository);
        
        
        // Tải cấu hình Node CỤ THỂ
        NodeInfo initialInfo = nodeRepository.getNodeInfo(nodeId);
        
        if (initialInfo == null) {
            logger.error("LỖI KHỞI CHẠY: Không tìm thấy cấu hình cho Node ID: {}. Dừng.", nodeId);
            return;
        }

        // --- 1B. XÁC ĐỊNH CỔNG VÀ CẬP NHẬT ĐỊA CHỈ MẠNG CỤC BỘ ---
        // Port được lấy từ DB (nếu có) hoặc dùng mặc định.
        int port = initialInfo.getPort() > 0 ? initialInfo.getPort() : FALLBACK_PORT;
        
        try {
            String hostName = getActualHostName(nodeId); 
            
            initialInfo.setHost(hostName);
            initialInfo.setPort(port); 
            
            nodeRepository.updateNodeInfo(nodeId, initialInfo); 
            logger.info("Địa chỉ mạng Node {} đã được đồng bộ lên DB: {}:{}", nodeId, hostName, port);
            
        } catch (Exception e) {
            logger.error("LỖI CẬP NHẬT ĐỊA CHỈ MẠNG: Không thể xác định Hostname. {}", e.getMessage());
            return;
        }
        
        // --- 2. KHỞI TẠO CÁC SERVICE CỐT LÕI (DI) ---
        ILinkManagerService linkManager = new LinkManagerService();
        IUserService userService = new UserService();
        IRoutingEngine routingEngine = new DijkstraRoutingEngine(); 
        
        NetworkManagerService networkManager = new NetworkManagerService(
            linkManager, 
            nodeRepository,
            routingEngine
        );
        
        // --- 3. KHỞI TẠO NODE VÀ GATEWAY (TCP LISTENER) ---
        
        INodeService nodeService = new NodeService(initialInfo, networkManager, userService, nodeRepository);
        INodeGatewayService gatewayService = new TcpGatewayService();
        gatewayService.setNodeServiceReference(nodeService);
        
        // Lắng nghe TCP thực sự trên cổng đã xác định
        gatewayService.startListening(initialInfo, port);
        
        // Bắt đầu vòng lặp xử lý của Node
        nodeService.startSimulationLoop();

        // --- 4. BẮT ĐẦU VÒNG LẶP ĐỊNH TUYẾN TOÀN MẠNG ---
        
        ServiceQoS baseQoS = userService.getQoSForPacket(createDummyDataPacket(nodeId, "DUMMY_DEST", "DATA")); 
        networkManager.startNetworkSimulation(nodeRepository.loadAllNodeConfigs(), baseQoS);
        
        logger.info("Node {} đã khởi động thành công và đang lắng nghe Client trên cổng {}.", nodeId, port);
    }
    
    // Hàm phụ trợ (Không thay đổi)
    private Packet createDummyDataPacket(String sourceId, String destId, String serviceType) {
        Packet packet = new Packet();
        packet.setPacketId("DUMMY_QOS_CHECK");
        packet.setType(Packet.PacketType.DATA);
        packet.setSourceUserId(sourceId);
        packet.setDestinationUserId(destId);
        packet.setServiceType(serviceType);
        return packet;
    }

    private void checkAndSeedDatabase(INodeRepository repository) {
        NodeSeeder seeder = new NodeSeeder(repository);
        seeder.seedInitialNodes(false); 
    }
    
    /**
     * Hàm lấy Hostname thực tế cho môi trường Docker/Local.
     */
    private String getActualHostName(String nodeId) throws UnknownHostException {
        // Sử dụng logic mặc định: Tên Service Docker (viết thường)
        return nodeId.toLowerCase(); 
    }
}
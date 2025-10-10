package com.sagin.util;

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
import com.sagin.repository.INodeRepository;
import com.sagin.repository.MongoNodeRepository;
import com.sagin.routing.DijkstraRoutingEngine;
import com.sagin.routing.IRoutingEngine;
import com.sagin.seeding.AsiaNodeSeeder;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

        // --- 1. KHỞI TẠO REPOSITORY VÀ SEEDING DỮ LIỆU ---
        INodeRepository nodeRepository = new MongoNodeRepository();
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
            String hostName = "localhost";
            
            logger.info("Địa chỉ mạng hiện tại của Node {}: {}:{}", nodeId, hostName, port);
            // Set host/port in-memory only. We intentionally do NOT persist the host
            // (or overwrite the DB copy) here to avoid seeding/runtime logic that
            // would set the host equal to the nodeId or otherwise leak local values
            // into the shared database.
            initialInfo.setHost(hostName);
            initialInfo.setPort(port);

            // NOTE: Do NOT call nodeRepository.updateNodeInfo(...) here. Persisting
            // the host during node startup can overwrite the intended DB-stored
            // network configuration (and previously was set from nodeId). Keep the
            // change local to this process only.
            logger.info("Địa chỉ mạng Node {} đã được cập nhật cục bộ: {}:{} (không ghi lên DB)", nodeId, hostName, port);
            
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
        AsiaNodeSeeder seeder = new AsiaNodeSeeder(repository);
        seeder.seedNodes(true); 
    }
    
    // /**
    //  * Hàm lấy Hostname thực tế cho môi trường Docker/Local.
    //  */
    // private String getActualHostName(String nodeId) throws UnknownHostException {
    //     return nodeId.toLowerCase(); 
    // }
}
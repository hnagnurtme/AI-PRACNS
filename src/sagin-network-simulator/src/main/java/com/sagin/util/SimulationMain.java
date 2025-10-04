package com.sagin.util;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeGatewayService;
import com.sagin.core.INodeService;
import com.sagin.core.IPacketService;
import com.sagin.core.service.NodeService;
import com.sagin.configuration.ServiceConfiguration;
import com.sagin.model.NodeInfo;
import com.sagin.core.ILinkManagerService; 
import com.sagin.routing.RoutingEngine;
import com.sagin.repository.INodeRepository; 
import com.sagin.seeding.NodeSeeder;       

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);

    public static void main(String[] args) {
        
        if (args.length < Initializer.REQUIRED_ARGS_COUNT) {
            logger.error("Lỗi: Thiếu tham số khởi tạo. Cần ít nhất {} tham số.", Initializer.REQUIRED_ARGS_COUNT);
            System.exit(1);
        }
        
        try {
            // 1. LẤY CẤU HÌNH DỊCH VỤ (SINGLETON)
            ServiceConfiguration config = ServiceConfiguration.getInstance();
            
            // 2. LẤY TẤT CẢ DEPENDENCY TỪ CONFIG
            INetworkManagerService networkManager = config.getNetworkManagerService();
            RoutingEngine routingEngine = config.getRoutingEngine();
            ILinkManagerService linkManager = config.getLinkManagerService();
            INodeRepository nodeRepository = config.getNodeRepository(); 
            IPacketService packetService = config.getPacketService();
            
            // ❗ LỖI SỬA: Lấy Gateway Service từ Configuration ❗
            INodeGatewayService nodeGateway = config.getNodeGatewayService(); 

            // 3. THỰC HIỆN SEEDING DỮ LIỆU
            NodeSeeder seeder = new NodeSeeder(nodeRepository);
            seeder.seedInitialNodes(false); 

            // 4. Khởi tạo Node Info
            NodeInfo currentNodeInfo = Initializer.initializeNodeFromArgs(args);
            
            // 5. Khởi tạo Node Service THỰC HIỆN DEPENDENCY INJECTION HOÀN CHỈNH
            INodeService nodeService = new NodeService( // Phải dùng tên lớp NodeService đã được sửa
                currentNodeInfo, 
                networkManager,   
                routingEngine,    
                linkManager,
                nodeGateway      
            );
            
            // 6. GIẢI QUYẾT VÒNG LẶP PHỤ THUỘC (Setter Injection)
            // TcpNodeGateway cần NodeService để đưa gói tin vào buffer
            nodeGateway.setNodeServiceReference(nodeService);

            logger.info("=================================================");
            logger.info("Node ID: {} | Type: {}", currentNodeInfo.getNodeId(), currentNodeInfo.getNodeType());
            logger.info("Vị trí: {}", currentNodeInfo.getPosition().toString());
            logger.info("BW Max: {} Mbps", currentNodeInfo.getCurrentBandwidth());
            logger.info("=================================================");

            // 7. Cấu hình ban đầu của Network Manager 
            Map<String, NodeInfo> currentInstanceConfig = new HashMap<>();
            currentInstanceConfig.put(currentNodeInfo.getNodeId(), currentNodeInfo);
            networkManager.initializeNetwork(currentInstanceConfig); 
            
            // 8. Đăng ký Node vào Registry và bắt đầu mô phỏng
            networkManager.registerActiveNode(currentNodeInfo.getNodeId(), nodeService);
            nodeService.startSimulationLoop(); 

        } catch (IllegalArgumentException e) {
            logger.error("Lỗi tham số khởi động: {}", e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            logger.error("Lỗi nghiêm trọng xảy ra trong quá trình khởi tạo ứng dụng:", e);
            System.exit(1);
        }
    }
}
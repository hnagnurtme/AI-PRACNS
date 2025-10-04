package com.sagin.util;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.core.service.NodeService;
import com.sagin.configuration.ServiceConfiguration;
import com.sagin.model.NodeInfo;
import com.sagin.core.ILinkManagerService;
import com.sagin.routing.RoutingEngine;
import com.sagin.repository.INodeRepository; // Cần thiết cho Seeder
import com.sagin.seeding.NodeSeeder;       // Import NodeSeeder

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Điểm khởi chạy chính của ứng dụng mô phỏng Node.
 * Lớp này thực hiện Dependency Injection và khởi tạo luồng mạng chính.
 */
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
            
            // 2. LẤY CÁC DEPENDENCY CẦN THIẾT
            INetworkManagerService networkManager = config.getNetworkManagerService();
            RoutingEngine routingEngine = config.getRoutingEngine();
            ILinkManagerService linkManager = config.getLinkManagerService();
            INodeRepository nodeRepository = config.getNodeRepository(); // 👈 Lấy Repository cho Seeder

            // 3. THỰC HIỆN SEEDING DỮ LIỆU
            NodeSeeder seeder = new NodeSeeder(nodeRepository);
            // Chạy Seeder: Đặt 'true' nếu muốn ghi đè Database mỗi lần chạy (dùng cho testing)
            seeder.seedInitialNodes(false); 

            // 4. Khởi tạo Node Info từ tham số dòng lệnh
            NodeInfo currentNodeInfo = Initializer.initializeNodeFromArgs(args);
            
            logger.info("=================================================");
            logger.info("Node ID: {} | Type: {}", currentNodeInfo.getNodeId(), currentNodeInfo.getNodeType());
            logger.info("Vị trí: {}", currentNodeInfo.getPosition().toString());
            logger.info("BW Max: {} Mbps", currentNodeInfo.getCurrentBandwidth());
            logger.info("=================================================");

            // 5. Cấu hình ban đầu của Network Manager 
            Map<String, NodeInfo> currentInstanceConfig = new HashMap<>();
            currentInstanceConfig.put(currentNodeInfo.getNodeId(), currentNodeInfo);
            
            // initializeNetwork sẽ tải dữ liệu từ DB (vừa được seeder đẩy lên) VÀ thêm Node hiện tại
            networkManager.initializeNetwork(currentInstanceConfig); 

            // 6. Khởi tạo Node Service (THỰC HIỆN DEPENDENCY INJECTION)
            INodeService nodeService = new NodeService(
                currentNodeInfo, 
                networkManager,
                routingEngine,   
                linkManager       
            );
            
            // 7. Đăng ký Node vào Registry và bắt đầu mô phỏng
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
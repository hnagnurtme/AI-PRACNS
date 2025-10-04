package com.sagin.util;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.core.service.NodeService;
import com.sagin.configuration.ServiceConfiguration; // Import Service Configuration
import com.sagin.model.Geo3D; // Cần cho createMockNetworkConfigs
import com.sagin.model.NodeInfo;
import com.sagin.core.ILinkManagerService; // Cần cho DI
import com.sagin.routing.RoutingEngine; // Cần cho DI

import java.util.HashMap;
import java.util.Map;

/**
 * Điểm khởi chạy chính của ứng dụng mô phỏng Node.
 * Sử dụng ServiceConfiguration để khởi tạo và kết nối tất cả các Service.
 */
public class SimulationMain {

    public static void main(String[] args) {
        
        if (args.length < Initializer.REQUIRED_ARGS_COUNT) {
            System.err.println("Lỗi: Thiếu tham số khởi tạo. Cần ít nhất " + Initializer.REQUIRED_ARGS_COUNT + " tham số.");
            System.exit(1);
        }
        
        try {
            // 1. LẤY CẤU HÌNH DỊCH VỤ (SINGLETON)
            ServiceConfiguration config = ServiceConfiguration.getInstance();
            
            // 2. LẤY CÁC DEPENDENCY CẦN THIẾT CHO INJECTION
            INetworkManagerService networkManager = config.getNetworkManagerService();
            RoutingEngine routingEngine = config.getRoutingEngine();
            ILinkManagerService linkManager = config.getLinkManagerService(); // LinkManagerService là implementation của ILinkManagerService

            // 3. Khởi tạo Node Info từ tham số dòng lệnh
            NodeInfo currentNodeInfo = Initializer.initializeNodeFromArgs(args);
            
            System.out.println("-------------------------------------------------");
            System.out.println("Node ID: " + currentNodeInfo.getNodeId() + 
                               " | Type: " + currentNodeInfo.getNodeType());
            System.out.println("Vị trí: " + currentNodeInfo.getPosition().toString());
            System.out.println("BW Max: " + currentNodeInfo.getCurrentBandwidth() + " Mbps");
            System.out.println("-------------------------------------------------");

            // 4. Cấu hình ban đầu của Network Manager (Tạo Database Vị trí)
            Map<String, NodeInfo> initialConfigs = createMockNetworkConfigs(currentNodeInfo);
            networkManager.initializeNetwork(initialConfigs); // Đăng ký tất cả node vào Manager

            // 5. Khởi tạo Node Service THỰC HIỆN DEPENDENCY INJECTION
            // Truyền tất cả các dependencies mà NodeService cần để hoạt động
            INodeService nodeService = new NodeService(
                currentNodeInfo, 
                networkManager,   // Dependency 1: INetworkManagerService
                routingEngine,    // Dependency 2: RoutingEngine
                linkManager       // Dependency 3: ILinkManagerService
            );
            
            // 6. Đăng ký Node vào Registry và bắt đầu mô phỏng
            networkManager.registerActiveNode(currentNodeInfo.getNodeId(), nodeService);
            nodeService.startSimulationLoop(); 

        } catch (IllegalArgumentException e) {
            System.err.println("Lỗi tham số: " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Lỗi nghiêm trọng xảy ra trong quá trình khởi tạo: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * Phương thức giả lập cấu hình Node cho toàn mạng (Database Vị trí ban đầu).
     */
    private static Map<String, NodeInfo> createMockNetworkConfigs(NodeInfo currentNodeInfo) {
        Map<String, NodeInfo> configs = new HashMap<>();
        
        // Thêm chính Node đang chạy vào configs
        configs.put(currentNodeInfo.getNodeId(), currentNodeInfo);
        
        // --- Thêm Node láng giềng tĩnh GS_001 ---
        NodeInfo gsNeighbor = new NodeInfo();
        gsNeighbor.setNodeId("GS_001");
        gsNeighbor.setNodeType(ProjectConstant.NODE_TYPE_GROUND_STATION);
        gsNeighbor.setPosition(new Geo3D(31.0, -101.0, 0.001));
        gsNeighbor.setCurrentBandwidth(5000.0);
        configs.put("GS_001", gsNeighbor);
        
        // --- Thêm Node láng giềng tĩnh SAT_001 ---
        // (Cần thiết để cả hai node có thể tìm thấy thông tin của nhau)
        NodeInfo satNeighbor = new NodeInfo();
        satNeighbor.setNodeId("SAT_001");
        satNeighbor.setNodeType(ProjectConstant.NODE_TYPE_SATELLITE);
        satNeighbor.setPosition(new Geo3D(30.0, -100.0, 550.0));
        satNeighbor.setCurrentBandwidth(1000.0);
        configs.put("SAT_001", satNeighbor);
        
        return configs;
    }
}
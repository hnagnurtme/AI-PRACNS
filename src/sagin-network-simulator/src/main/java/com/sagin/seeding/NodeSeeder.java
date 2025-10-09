package com.sagin.seeding;

import com.sagin.repository.INodeRepository;
import com.sagin.model.*; 
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Lớp Seeder chịu trách nhiệm khởi tạo 20 Node cấu hình mẫu cho mạng SAGSINS.
 * Đảm bảo các Node có vị trí chiến lược và trạng thái hoạt động.
 */
public class NodeSeeder {

    private static final Logger logger = LoggerFactory.getLogger(NodeSeeder.class);
    private final INodeRepository nodeRepository;
    private final Random random = new Random();
    private static final int BASE_PORT = 8080; // Cổng khởi đầu cho các node

    public NodeSeeder(INodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    /**
     * Tạo dữ liệu mẫu và đẩy lên Repository/Database.
     * @param forceOverwrite Nếu true, sẽ ghi đè dữ liệu hiện có.
     */
    public void seedInitialNodes(boolean forceOverwrite) {
        logger.info("Bắt đầu quy trình gieo mầm (seeding) Node cấu hình.");
        
        Map<String, NodeInfo> defaultNodes = createDefaultNodes();
        
        // 2. Kiểm tra trạng thái hiện tại
        Map<String, NodeInfo> currentNodes = nodeRepository.loadAllNodeConfigs();

        if (!currentNodes.isEmpty() && !forceOverwrite) {
            logger.warn("Database đã có {} Node. Bỏ qua quy trình seeding.", currentNodes.size());
            return;
        }

        // 3. Đẩy dữ liệu lên Database
        defaultNodes.forEach((id, info) -> {
            nodeRepository.updateNodeInfo(id, info);
        });

        logger.info("Đã thêm/cập nhật {} Node cấu hình mẫu vào Database.", defaultNodes.size());
    }

    /**
     * Tạo 20 đối tượng NodeInfo mẫu phản ánh thực tế SAGSINS.
     */
    private Map<String, NodeInfo> createDefaultNodes() {
        Map<String, NodeInfo> nodes = new HashMap<>();
        long now = System.currentTimeMillis();
        
        final int NUM_LEO = 12;
        final int NUM_MEO = 3;
        int portCounter = BASE_PORT;
        
        // --- 1. TRẠM MẶT ĐẤT (4 GS) ---
        nodes.put("GS_TOKYO", createGroundStation("GS_TOKYO", 35.68, 139.69, 0.01, WeatherCondition.CLEAR, 1.0, portCounter++));
        nodes.put("GS_LONDON", createGroundStation("GS_LONDON", 51.50, -0.12, 0.02, WeatherCondition.RAIN, 1.5, portCounter++));
        nodes.put("GS_NY", createGroundStation("GS_NY", 40.71, -74.00, 0.05, WeatherCondition.SNOW, 1.2, portCounter++));
        nodes.put("GS_SYDNEY", createGroundStation("GS_SYDNEY", -33.86, 151.20, 0.03, WeatherCondition.LIGHT_RAIN, 1.1, portCounter++));
        
        // --- 2. VỆ TINH LEO (12 LEO) ---
        for (int i = 1; i <= NUM_LEO; i++) {
            String nodeId = String.format("LEO_%03d", i);
            double inclination = (i <= 6) ? 53.0 : 85.0;
            double altitude = (i <= 6) ? 550.0 : 1200.0;
            
            // LEO_001: Vị trí chiến lược (BUỘC KẾT NỐI GS_LONDON)
            if (i == 1) {
                 nodes.put(nodeId, createSatellite(
                    nodeId, NodeType.LEO_SATELLITE, 
                    40.0, 5.0, altitude, // Vị trí trên Tây Âu, gần London
                    7.6, 0.5, 0.0,
                    90.0, 80, inclination, now, portCounter++
                ));
            }
            // LEO_012: Vị trí chiến lược (Gần Châu Á/Úc) để kết nối GS_SYDNEY
            else if (i == 12) {
                 nodes.put(nodeId, createSatellite(
                    nodeId, NodeType.LEO_SATELLITE, 
                    -20.0, 150.0, altitude, 
                    7.2, 1.0, 0.0,
                    90.0, 100, inclination, now, portCounter++
                ));
            } 
            // Các LEO khác (Ngẫu nhiên)
            else {
                nodes.put(nodeId, createSatellite(
                    nodeId, NodeType.LEO_SATELLITE, 
                    random.nextDouble() * 100 - 50, random.nextDouble() * 180 - 90, altitude, 
                    7.5, random.nextDouble() * 0.5, 0.0,
                    80.0 + random.nextDouble() * 10, 80, inclination, now, portCounter++
                ));
            }
        }
        
        // --- 3. VỆ TINH MEO (3 MEO) ---
        for (int i = 1; i <= NUM_MEO; i++) {
            String nodeId = String.format("MEO_%03d", i);
            nodes.put(nodeId, createSatellite(
                nodeId, NodeType.MEO_SATELLITE, 
                random.nextDouble() * 60 - 30, random.nextDouble() * 180 - 90, 15000.0, 
                3.0 + random.nextDouble() * 0.5, 0.0, 0.0, 
                90.0 + random.nextDouble() * 5, 150, 60.0, now, portCounter++
            ));
        }
        
        // --- 4. VỆ TINH GEO (1 GEO) ---
        nodes.put("GEO_001", createSatellite(
            "GEO_001", 
            NodeType.GEO_SATELLITE, 
            0.0, -100.0, 35786.0, 
            0.001, 0.001, 0.001, 
            99.9, 200, 0.0, now, portCounter++
        ));
        
        return nodes; // Tổng cộng 20 Nodes
    }

    // --- Phương thức Khởi tạo Mẫu (Đảm bảo logic hoạt động) ---

    private NodeInfo createGroundStation(String id, double lat, double lon, double alt, WeatherCondition weather, double delayMs, int port) {
        NodeInfo info = new NodeInfo();
        info.setNodeId(id);
        info.setNodeType(NodeType.GROUND_STATION);
        info.setPosition(new Geo3D(lat, lon, alt));
        
    // Cấu hình mạng (Host sẽ được cập nhật khi Node khởi chạy)
    // NOTE: Do NOT auto-assign host from the node ID during seeding. Host
    // should remain unset here so it isn't written to the DB as the node ID.
        info.setPort(port);
        
        info.setOperational(true); // BẮT BUỘC: Node phải hoạt động
        
        info.setBatteryChargePercent(100.0);
        info.setNodeProcessingDelayMs(delayMs);
        info.setPacketLossRate(0.00001); 
        info.setPacketBufferCapacity(500);
        info.setCurrentPacketCount(random.nextInt(50));
        info.setResourceUtilization(random.nextDouble() * 0.2);
        info.setWeather(weather);
        
        info.setOrbit(null); 
        info.setVelocity(null);
        
        info.setLastUpdated(System.currentTimeMillis());
        return info;
    }

    private NodeInfo createSatellite(String id, NodeType type, double lat, double lon, double alt, double vx, double vy, double vz, double battery, int bufferCapacity, double inclinationDeg, long timestamp, int port) {
        NodeInfo info = new NodeInfo();
        info.setNodeId(id);
        info.setNodeType(type);
        info.setPosition(new Geo3D(lat, lon, alt));
        info.setVelocity(new Velocity(vx, vy, vz));
        
    // Cấu hình mạng
    // NOTE: Do NOT auto-assign host from the node ID during seeding. Host
    // should remain unset here so it isn't written to the DB as the node ID.
        info.setPort(port);
        
        info.setOrbit(new Orbit(alt + 6371.0, 0.001, inclinationDeg, random.nextDouble() * 360, 0.0, 0.0)); 
        
        info.setOperational(true); 
        info.setBatteryChargePercent(battery);
        info.setNodeProcessingDelayMs(0.8);
        info.setPacketLossRate(0.005);
        info.setPacketBufferCapacity(bufferCapacity);
        info.setCurrentPacketCount(random.nextInt(bufferCapacity / 5));
        info.setResourceUtilization(random.nextDouble() * 0.4);
        info.setWeather(WeatherCondition.CLEAR);
        info.setLastUpdated(timestamp);
        return info;
    }
}
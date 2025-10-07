package com.sagin.seeding;

import com.sagin.repository.INodeRepository;
import com.sagin.model.NodeInfo;
import com.sagin.model.Geo3D;
import com.sagin.util.ProjectConstant;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Lớp Seeder chịu trách nhiệm khởi tạo dữ liệu cấu hình Node ban đầu (NodeInfo) 
 * lên Database (Firestore).
 * Việc này giúp đảm bảo mạng lưới có sẵn các Node cố định để hoạt động.
 */
public class NodeSeeder {

    private static final Logger logger = LoggerFactory.getLogger(NodeSeeder.class);
    private final INodeRepository nodeRepository;

    public NodeSeeder(INodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    /**
     * Tạo dữ liệu mẫu và đẩy lên Repository/Database.
     * @param forceOverwrite Nếu true, sẽ ghi đè dữ liệu hiện có.
     */
    public void seedInitialNodes(boolean forceOverwrite) {
        logger.info("Bắt đầu quy trình gieo mầm (seeding) Node cấu hình.");
        
        // 1. Tạo danh sách các NodeInfo mẫu
        Map<String, NodeInfo> defaultNodes = createDefaultNodes();
        
        // 2. Kiểm tra trạng thái hiện tại (Kiểm tra xem Database đã có Node chưa)
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
     * Tạo các đối tượng NodeInfo mẫu (dùng cho seeding).
     */
    private Map<String, NodeInfo> createDefaultNodes() {
    Map<String, NodeInfo> nodes = new HashMap<>();

    long currentTime = System.currentTimeMillis();

    // 1. Node Vệ tinh LEO (SAT_001)
    NodeInfo sat001 = new NodeInfo();
    sat001.setNodeId("SAT_001");
    sat001.setNodeType(ProjectConstant.NODE_TYPE_SATELLITE);
    sat001.setPosition(new Geo3D(30.0, -100.0, 550.0));
    
    // Trạng thái QoS
    sat001.setOperational(true);
    sat001.setCurrentBandwidth(1000.0);       // Mbps
    sat001.setAvgLatencyMs(15.0);             // ms (Độ trễ truyền dẫn cao do khoảng cách)
    sat001.setPacketLossRate(0.02);           // 2%
    
    // Tài nguyên & Hiệu suất
    sat001.setPacketBufferLoad(0);            
    sat001.setCurrentThroughput(0.0);
    sat001.setResourceUtilization(0.15);       // 15% CPU/Memory
    sat001.setPowerLevel(100.0);               // Nguồn cấp ổn định
    sat001.setLastUpdated(currentTime);
    nodes.put("SAT_001", sat001);

    // 2. Node Trạm Mặt đất (GS_001)
    NodeInfo gs001 = new NodeInfo();
    gs001.setNodeId("GS_001");
    gs001.setNodeType(ProjectConstant.NODE_TYPE_GROUND_STATION);
    gs001.setPosition(new Geo3D(31.0, -101.0, 0.001)); 
    
    // Trạng thái QoS
    gs001.setOperational(true);
    gs001.setCurrentBandwidth(5000.0);        // Mbps (Băng thông rất lớn)
    gs001.setAvgLatencyMs(1.0);               // ms (Độ trễ gần như tức thời)
    gs001.setPacketLossRate(0.005);           // 0.5% (Rất ổn định)
    
    // Tài nguyên & Hiệu suất
    gs001.setPacketBufferLoad(0);
    gs001.setCurrentThroughput(0.0);
    gs001.setResourceUtilization(0.05);       // 5% CPU/Memory
    gs001.setPowerLevel(100.0);               // Nguồn cấp ổn định
    gs001.setLastUpdated(currentTime);
    nodes.put("GS_001", gs001);
    
    // 3. Node Tàu biển/Trạm biển (SEA_001)
    NodeInfo sea001 = new NodeInfo();
    sea001.setNodeId("SEA_001");
    sea001.setNodeType(ProjectConstant.NODE_TYPE_SEA_VESSEL);
    sea001.setPosition(new Geo3D(28.0, -102.0, 0.0));
    
    // Trạng thái QoS
    sea001.setOperational(true);
    sea001.setCurrentBandwidth(500.0);
    sea001.setAvgLatencyMs(10.0);             
    sea001.setPacketLossRate(0.05);           // 5% (Tỷ lệ mất gói cao hơn)
    
    // Tài nguyên & Hiệu suất
    sea001.setPacketBufferLoad(0);
    sea001.setCurrentThroughput(0.0);
    sea001.setResourceUtilization(0.10);
    sea001.setPowerLevel(90.0);               // Giả định dùng pin/nguồn hạn chế
    sea001.setLastUpdated(currentTime);
    nodes.put("SEA_001", sea001);

    return nodes;
}

}
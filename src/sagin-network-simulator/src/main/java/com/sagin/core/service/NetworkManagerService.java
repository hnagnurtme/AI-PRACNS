package com.sagin.core.service;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.repository.INodeRepository; 

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lớp triển khai INetworkManagerService. 
 * Quản lý và điều phối tất cả NodeService đang chạy.
 * Lớp này hoạt động như một Registry và API Gateway cho mạng mô phỏng.
 */
public class NetworkManagerService implements INetworkManagerService {

    private static final Logger logger = LoggerFactory.getLogger(NetworkManagerService.class);

    // Lưu trữ tất cả NodeService đang hoạt động (Node ID -> NodeService Object)
    private final Map<String, INodeService> activeNodeServices;
    // Lưu trữ tất cả NodeInfo (Database Vị trí trong bộ nhớ)
    private final Map<String, NodeInfo> networkNodesInfo;
    
    // DEPENDENCY: Repository để tải dữ liệu từ DB
    private final INodeRepository nodeRepository; 

    public NetworkManagerService(INodeRepository nodeRepository) { // 👈 SỬA: Nhận Repository
        this.activeNodeServices = new ConcurrentHashMap<>();
        this.networkNodesInfo = new ConcurrentHashMap<>();
        this.nodeRepository = nodeRepository; 
        logger.info("NetworkManagerService đã khởi tạo.");
    }

    @Override
    public void initializeNetwork(Map<String, NodeInfo> initialNodeConfigs) {
        logger.info("Khởi tạo cấu trúc mạng: Bắt đầu tải dữ liệu Node...");
        
        Map<String, NodeInfo> dbConfigs = nodeRepository.loadAllNodeConfigs();
        
        this.networkNodesInfo.putAll(dbConfigs);
        
        this.networkNodesInfo.putAll(initialNodeConfigs);

        logger.info("Tải thành công {} Node (Bao gồm cả Node đang chạy) vào Registry.", 
                    this.networkNodesInfo.size());
    }
    
    @Override
    public void registerActiveNode(String serviceId, INodeService nodeService) {
        if (!activeNodeServices.containsKey(serviceId)) {
            activeNodeServices.put(serviceId, nodeService);
            logger.info("Node {} đã đăng ký thành công vào NetworkManager.", serviceId);
        } else {
            logger.warn("Node {} đã tồn tại trong danh sách Node hoạt động (Đã đăng ký lại).", serviceId);
        }
    }

    @Override
    public void transferPacket(Packet packet, String destNodeId) {
        INodeService destinationNode = activeNodeServices.get(destNodeId);
        
        if (destinationNode != null) {
            // Gọi phương thức receivePacket() của Node đích
            logger.info("Chuyển giao: Gói {} từ {} -> {}", 
                        packet.getPacketId(), packet.getCurrentHoldingNodeId(), destNodeId);
            // Kỹ thuật gọi hàm này là cách mô phỏng Network Hand-off giữa các luồng
            destinationNode.receivePacket(packet);
            
            // NOTE: Cần có logic cập nhật vị trí/trạng thái lên DB tại đây nếu dùng Firebase
            // nodeRepository.updateNodeInfo(packet.getCurrentHoldingNodeId(), latestNodeInfo);
        } else {
            logger.warn("LỖI CHUYỂN GIAO: Node đích {} không tồn tại hoặc không hoạt động.", destNodeId);
            packet.markDropped();
        }
    }

    @Override
    public NodeInfo getNodeInfo(String nodeId) {
        // Cung cấp thông tin của các node khác trong mạng (cho khám phá láng giềng)
        return networkNodesInfo.get(nodeId);
    }

    @Override
    public void startSimulation() {
        logger.info("Network Manager đã sẵn sàng.");
    }
}
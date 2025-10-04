package com.sagin.core.service;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lớp triển khai INetworkManagerService. 
 * Quản lý và điều phối tất cả NodeService đang chạy.
 */
public class NetworkManagerService implements INetworkManagerService {

    private static final Logger logger = LoggerFactory.getLogger(NetworkManagerService.class);

    // Lưu trữ tất cả NodeService đang hoạt động (Node ID -> NodeService Object)
    private final Map<String, INodeService> activeNodeServices;
    // Lưu trữ tất cả NodeInfo (Node ID -> NodeInfo Object)
    private final Map<String, NodeInfo> networkNodesInfo;

    public NetworkManagerService() {
        this.activeNodeServices = new ConcurrentHashMap<>();
        this.networkNodesInfo = new ConcurrentHashMap<>();
    }

    @Override
    public void initializeNetwork(Map<String, NodeInfo> initialNodeConfigs) {
        logger.info("Khởi tạo mạng lưới với {} node...", initialNodeConfigs.size());
        
        for (NodeInfo info : initialNodeConfigs.values()) {
            // Lưu NodeInfo vào Map chung
            this.networkNodesInfo.put(info.getNodeId(), info);
            
            // Khởi tạo NodeService cho từng Node
            // LƯU Ý: NetworkManager thường khởi tạo và quản lý NodeService
            // Nhưng trong mô hình Docker Compose, mỗi container tự khởi tạo NodeService của mình
            // Ở đây, ta chỉ giả định rằng nó lưu trữ các NodeInfo
        }
        
        logger.info("Khởi tạo cấu trúc liên kết mạng hoàn tất.");
    }

    @Override
    public void transferPacket(Packet packet, String destNodeId) {
        INodeService destinationNode = activeNodeServices.get(destNodeId);
        
        if (destinationNode != null) {
            // Gọi phương thức receivePacket() của Node đích
            logger.info("Chuyển giao: Gói {} từ {} -> {}", 
                        packet.getPacketId(), packet.getCurrentHoldingNodeId(), destNodeId);
            destinationNode.receivePacket(packet);
        } else {
            logger.warn("LỖI CHUYỂN GIAO: Node đích {} không tồn tại hoặc không hoạt động.", destNodeId);
            // Xử lý gói tin bị mất
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
        logger.info("Bắt đầu vòng lặp thời gian toàn mạng.");
        // Giai đoạn này thường bao gồm việc đồng bộ hóa các NodeService đã được khởi động
        // (Trong mô hình Docker, NodeService tự chạy, NetworkManager chỉ đóng vai trò là Registry/API Gateway)
        
        // Logic mô phỏng: Kích hoạt tất cả NodeService (nếu chưa chạy)
        // for (INodeService service : activeNodeServices.values()) {
        //     service.startSimulationLoop();
        // }
    }

    @Override
    public void registerActiveNode(String serviceId, INodeService nodeService) {
        if (!activeNodeServices.containsKey(serviceId)) {
            activeNodeServices.put(serviceId, nodeService);
            logger.info("Node {} đã đăng ký thành công vào NetworkManager.", serviceId);
        } else {
            logger.warn("Node {} đã tồn tại trong danh sách Node hoạt động.", serviceId);
        }
    }

}
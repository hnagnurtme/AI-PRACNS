package com.sagin.core;

import com.sagin.model.Packet;
import com.sagin.model.NodeInfo;
import java.util.Map;

/**
 * Quản lý và Điều phối toàn bộ mạng lưới (Topology, Node Registry, Giao tiếp).
 */
public interface INetworkManagerService {

    /** Khởi tạo toàn bộ mạng lưới từ các cấu hình ban đầu. */
    void initializeNetwork(Map<String, NodeInfo> initialNodeConfigs);

    /**
     * Đăng ký một Node Service vừa được khởi tạo.
     * @param nodeId ID Node.
     * @param nodeService Đối tượng INodeService đã khởi động.
     */
    void registerActiveNode(String nodeId, INodeService nodeService);

    /**
     * Chuyển gói tin từ node gửi đến node đích kế tiếp.
     * Phương thức này mô phỏng việc truyền qua link và tính toán LinkMetric thực tế.
     * @param packet Gói tin cần chuyển (chứa nextHopNodeId).
     * @param sourceNodeId ID Node gửi gói tin đi.
     */
    void transferPacket(Packet packet, String sourceNodeId);

    /**
     * Lấy thông tin NodeInfo của một node khác trong mạng.
     * @param nodeId ID của node cần tra cứu.
     * @return NodeInfo của node đó.
     */
    NodeInfo getNodeInfo(String nodeId);
    
    /**
     * Lấy tất cả thông tin NodeInfo của các node đang hoạt động.
     */
    Map<String, NodeInfo> getAllNodeInfos();
}
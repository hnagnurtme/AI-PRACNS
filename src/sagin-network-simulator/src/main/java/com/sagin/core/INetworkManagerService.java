package com.sagin.core;

import com.sagin.model.Packet;
import com.sagin.model.NodeInfo;

import java.util.Map;

/**
 * Interface cho dịch vụ Quản lý và Điều phối toàn bộ mạng lưới (SAGINS).
 * Chịu trách nhiệm về cấu trúc liên kết và giao tiếp giữa các node.
 */
public interface INetworkManagerService {

    /** Khởi tạo toàn bộ mạng lưới (tạo tất cả NodeService). */
    void initializeNetwork(Map<String, NodeInfo> initialNodeConfigs);

     /**
     * Đăng ký một Node Service vừa được khởi tạo vào danh sách các Node đang hoạt động.
     * Đây là bước quan trọng để NetworkManager biết phải gọi receivePacket() của Node nào.
     * @param serviceId ID Node.
     * @param nodeService Đối tượng INodeService đã khởi động.
     */
    void registerActiveNode(String serviceId, INodeService nodeService); // 👈 ĐÃ BỔ SUNG

    /**
     * Chuyển gói tin từ node gửi đến node đích (Network hand-off).
     * Đây là cầu nối giữa các NodeService đang chạy trong các luồng/container khác nhau.
     * @param packet Gói tin cần chuyển.
     * @param destNodeId ID Node đích kế tiếp (Next Hop).
     */
    void transferPacket(Packet packet, String destNodeId);

    /**
     * Lấy thông tin NodeInfo của một node khác trong mạng (dùng cho khám phá láng giềng).
     * @param nodeId ID của node cần tra cứu.
     * @return NodeInfo của node đó.
     */
    NodeInfo getNodeInfo(String nodeId);

    /** Bắt đầu vòng lặp thời gian toàn mạng (Simulated Clock). */
    void startSimulation();
}
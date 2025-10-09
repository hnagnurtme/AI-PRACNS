package com.sagin.core;

import com.sagin.model.Packet;
import com.sagin.model.RouteInfo;
import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;

/**
 * Trái tim của mỗi node: Quản lý vòng lặp mô phỏng, xử lý gói tin, và định tuyến cục bộ.
 */
public interface INodeService {

    /** Khởi tạo và bắt đầu luồng xử lý chính cho Node. */
    void startSimulationLoop();

    /** * Nhận gói tin đến từ một node láng giềng.
     * @param packet Gói tin được gửi đến.
     * @param incomingLinkMetric LinkMetric của liên kết vừa truyền gói tin qua.
     */
    void receivePacket(Packet packet, LinkMetric incomingLinkMetric);

    /** * Cập nhật thông tin định tuyến (RouteInfo) cho một đích đến cụ thể.
     * Được gọi bởi RoutingEngine hoặc NetworkManager.
     */
    void updateRoute(String destinationId, RouteInfo routeInfo);

    /** Gửi gói tin đã được định tuyến đến node kế tiếp. */
    void sendPacket(Packet packet);

    /** Cập nhật định kỳ trạng thái node (vị trí, tài nguyên, sức khỏe). */
    void updateNodeState();
    
    /** * Lấy NodeInfo hiện tại của node này. 
     * Hữu ích cho các service khác tra cứu trạng thái cục bộ.
     */
    NodeInfo getNodeInfo();
}
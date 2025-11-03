package com.sagin.service;

import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;

import java.util.List;
import java.util.Map;

public interface INodeService {

    /**
     * Hạch toán chi phí NHẬN (RX) và XỬ LÝ (CPU) khi packet đến.
     * Hàm này được gọi đồng bộ (synchronously) bởi TCPServerListener.
     *
     * @param nodeId Node nhận packet
     * @param packet Packet vừa được nhận
     */
    void updateNodeStatus(String nodeId, Packet packet);

    /**
     * Hạch toán chi phí GỬI (TX) cho một lần truyền thành công.
     * <p>
     * Hàm này được gọi BẤT ĐỒNG BỘ (asynchronously) bởi TCP_Service
     * SAU KHI gửi qua socket được xác nhận là thành công.
     *
     * @param nodeId The ID of the node that sent the packet.
     * @param packet The packet that was successfully sent.
     */
    void processSuccessfulSend(String nodeId, Packet packet);

    /**
     * Xử lý một "tick" mô phỏng (hiện không dùng nhiều).
     */
    void processTick(Map<String, NodeInfo> nodeMap, List<Packet> incomingPackets);

    /**
     * Đẩy tất cả các node "dirty" (thay đổi) trong cache lên CSDL.
     */

    List<NodeInfo> getVisibleNodes(NodeInfo node, List<NodeInfo> allNodes);


    void updateNodeIpAddress(String nodeId, String newIpAddress);
    
    void flushToDatabase();
}
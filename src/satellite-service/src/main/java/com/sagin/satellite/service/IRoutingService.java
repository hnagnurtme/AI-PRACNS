package com.sagin.satellite.service;

import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.RoutingTable;
import com.sagin.satellite.model.LinkMetric;
import java.util.List;
import java.util.Map;

/**
 * IRoutingService định nghĩa các phương thức quản lý routing cho vệ tinh.
 * Tính toán đường đi tối ưu dựa trên các metric như khoảng cách, băng thông, độ trễ.
 */
public interface IRoutingService {

    /**
     * Tính toán routing table dựa trên topology hiện tại
     *
     * @param currentNodeId ID của node hiện tại
     * @param networkNodes Danh sách tất cả node trong mạng
     * @param linkMetrics Map các link metric giữa các node
     * @return RoutingTable đã tính toán
     */
    RoutingTable calculateRoutingTable(String currentNodeId, 
                                     List<NodeInfo> networkNodes, 
                                     Map<String, LinkMetric> linkMetrics);

    /**
     * Tìm next hop tới destination
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @return Node ID của next hop, null nếu không tìm thấy
     */
    String findNextHop(String sourceNodeId, String destinationNodeId);

    /**
     * Tìm đường đi tối ưu từ source tới destination
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @param networkNodes Danh sách node trong mạng
     * @param linkMetrics Map link metrics
     * @return Danh sách node ID tạo thành đường đi
     */
    List<String> findOptimalPath(String sourceNodeId, String destinationNodeId,
                               List<NodeInfo> networkNodes,
                               Map<String, LinkMetric> linkMetrics);

    /**
     * Cập nhật routing table khi có thay đổi topology
     *
     * @param routingTable Routing table cần cập nhật
     * @param updatedLinkMetrics Link metrics mới
     */
    void updateRoutingTable(RoutingTable routingTable, Map<String, LinkMetric> updatedLinkMetrics);

    /**
     * Kiểm tra xem có đường đi tới destination không
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @return true nếu có đường đi, false nếu không
     */
    boolean hasRouteTo(String sourceNodeId, String destinationNodeId);
}
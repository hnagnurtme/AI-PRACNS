package com.sagin.routing;

/**
 * Interface cho Dịch vụ Định tuyến.
 * Cung cấp logic để tìm đường đi tốt nhất cho một packet tại một node cụ thể.
 */
public interface IRoutingService {

    /**
     * Tìm RouteInfo (tuyến đường) tốt nhất cho một packet tại một node.
     *
     * @param currentNodeId   Node hiện tại đang giữ packet.
     * @param destinationNodeId Đích đến cuối cùng (ví dụ: một trạm mặt đất khác).
     * @return RouteInfo chứa thông tin nextHop, cost, v.v., hoặc null nếu không tìm thấy.
     */
    RouteInfo getBestRoute(String currentNodeId, String destinationNodeId);

    /**
     * Lấy bảng định tuyến (chỉ để đọc) của một node cụ thể.
     * @param nodeId ID của node.
     * @return Đối tượng RoutingTable.
     */
    RoutingTable getRoutingTableForNode(String nodeId);
    
    /**
     * Cập nhật một tuyến đường cho một node cụ thể (ví dụ: từ RL Agent).
     * @param forNodeId ID của node sở hữu bảng định tuyến.
     * @param newRoute Thông tin tuyến đường mới.
     */
    void updateRoute(String forNodeId, RouteInfo newRoute);
}
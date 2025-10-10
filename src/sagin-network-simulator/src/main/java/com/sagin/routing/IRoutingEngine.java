package com.sagin.routing;
import com.sagin.model.*;
import java.util.Map;

/**
 * Interface cho dịch vụ Tính toán Tuyến đường (Routing Engine).
 * Chịu trách nhiệm thực hiện các thuật toán định tuyến dựa trên QoS và Topology.
 */
public interface IRoutingEngine {

    /**
     * Tính toán toàn bộ bảng định tuyến (RoutingTable) cho node gốc (sourceNode).
     * * @param sourceNode Thông tin node đang tính toán bảng định tuyến (node hiện tại).
     * @param allActiveLinks Map chứa ID Link và LinkMetric của tất cả các link hoạt động trong mạng (Topology).
     * @param allNodeInfos Map chứa ID Node và NodeInfo của tất cả các node trong mạng.
     * @param targetQoS ServiceQoS mà tuyến đường cần tối ưu (ví dụ: Voice, Video).
     * @return Bảng định tuyến (RoutingTable) đã được tính toán.
     */
    RoutingTable computeRoutes(
        NodeInfo sourceNode, 
        Map<String, LinkMetric> allActiveLinks,
        Map<String, NodeInfo> allNodeInfos,
        ServiceQoS targetQoS
    );

}
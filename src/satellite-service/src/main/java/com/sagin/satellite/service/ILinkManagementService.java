package com.sagin.satellite.service;

import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.LinkMetric;
import java.util.List;
import java.util.Map;

/**
 * ILinkManagementService quản lý các liên kết giữa các node trong mạng vệ tinh.
 * Theo dõi chất lượng đường link, cập nhật metrics và phát hiện thay đổi topology.
 */
public interface ILinkManagementService {

    /**
     * Thiết lập link giữa hai node
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @param sourceNode Thông tin node nguồn
     * @param destinationNode Thông tin node đích
     * @return LinkMetric đã tạo
     */
    LinkMetric establishLink(String sourceNodeId, String destinationNodeId, 
                           NodeInfo sourceNode, NodeInfo destinationNode);

    /**
     * Cập nhật metrics của một link
     *
     * @param linkId ID của link (format: sourceId-destinationId)
     * @param bandwidthMbps Băng thông mới
     * @param latencyMs Độ trễ mới
     * @param packetLossRate Tỷ lệ mất packet mới
     * @param isAvailable Trạng thái khả dụng
     */
    void updateLinkMetrics(String linkId, double bandwidthMbps, double latencyMs, 
                          double packetLossRate, boolean isAvailable);

    /**
     * Lấy tất cả link metrics hiện tại
     *
     * @return Map với key là linkId, value là LinkMetric
     */
    Map<String, LinkMetric> getAllLinkMetrics();

    /**
     * Lấy link metric giữa hai node cụ thể
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @return LinkMetric nếu tồn tại, null nếu không
     */
    LinkMetric getLinkMetric(String sourceNodeId, String destinationNodeId);

    /**
     * Kiểm tra link giữa hai node có khả dụng không
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @return true nếu link khả dụng, false nếu không
     */
    boolean isLinkAvailable(String sourceNodeId, String destinationNodeId);

    /**
     * Lấy danh sách các node kề của một node
     *
     * @param nodeId ID của node
     * @return Danh sách ID các node kề
     */
    List<String> getNeighborNodes(String nodeId);

    /**
     * Xóa link giữa hai node
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     */
    void removeLink(String sourceNodeId, String destinationNodeId);

    /**
     * Cập nhật tất cả link distances dựa trên vị trí node mới
     *
     * @param updatedNodes Danh sách node đã cập nhật vị trí
     */
    void updateLinkDistances(List<NodeInfo> updatedNodes);

    /**
     * Phát hiện và loại bỏ các link không khả dụng
     *
     * @return Số lượng link đã bị loại bỏ
     */
    int cleanupDeadLinks();
}
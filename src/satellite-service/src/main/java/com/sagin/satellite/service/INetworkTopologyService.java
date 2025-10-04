package com.sagin.satellite.service;

import com.sagin.satellite.model.NodeInfo;
import java.util.List;
import java.util.Map;

/**
 * INetworkTopologyService quản lý topology của mạng vệ tinh
 */
public interface INetworkTopologyService {

    /**
     * Đăng ký node mới vào mạng
     *
     * @param nodeInfo Thông tin node mới
     */
    void registerNode(NodeInfo nodeInfo);

    /**
     * Hủy đăng ký node khỏi mạng
     *
     * @param nodeId ID của node cần hủy
     */
    void unregisterNode(String nodeId);

    /**
     * Cập nhật thông tin node
     *
     * @param nodeInfo Thông tin node đã cập nhật
     */
    void updateNode(NodeInfo nodeInfo);

    /**
     * Lấy thông tin tất cả nodes trong mạng
     *
     * @return Danh sách NodeInfo
     */
    List<NodeInfo> getAllNodes();

    /**
     * Lấy thông tin node cụ thể
     *
     * @param nodeId ID của node
     * @return NodeInfo nếu tồn tại, null nếu không
     */
    NodeInfo getNode(String nodeId);

    /**
     * Tìm các node kề của một node
     *
     * @param nodeId ID của node
     * @param maxDistance Khoảng cách tối đa (km)
     * @return Danh sách node kề
     */
    List<NodeInfo> findNeighborNodes(String nodeId, double maxDistance);

    /**
     * Cập nhật topology dựa trên vị trí mới của các node
     * Sẽ tự động tính toán lại các links có thể có
     *
     * @param maxLinkDistance Khoảng cách tối đa để tạo link (km)
     * @return Số lượng links mới được tạo
     */
    int updateTopology(double maxLinkDistance);

    /**
     * Lấy snapshot hiện tại của toàn bộ mạng
     *
     * @return Map chứa topology information
     */
    Map<String, Object> getNetworkSnapshot();

    /**
     * Kiểm tra connectivity giữa hai node
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @return true nếu có kết nối (trực tiếp hoặc gián tiếp)
     */
    boolean isConnected(String sourceNodeId, String destinationNodeId);

    /**
     * Tính toán độ phủ sóng của mạng
     *
     * @return Percentage coverage (0.0 - 1.0)
     */
    double calculateNetworkCoverage();

    /**
     * Phát hiện các node bị isolated
     *
     * @return Danh sách ID của các node không có kết nối
     */
    List<String> findIsolatedNodes();
}
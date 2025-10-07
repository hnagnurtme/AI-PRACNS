package com.sagin.routing;

import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.RoutingTable;
import java.util.Map;

/**
 * Interface cho các thuật toán định tuyến khác nhau (Dijkstra, RL, v.v.).
 * Định nghĩa hợp đồng cho việc tính toán và tra cứu tuyến đường.
 */
public interface RoutingEngine {

    /**
     * Tính toán toàn bộ bảng định tuyến (RoutingTable) cho node hiện tại.
     * Phương thức này là cốt lõi, sử dụng các thuật toán như Dijkstra hoặc kết quả từ RL.
     * @param currentNode Thông tin của node hiện tại.
     * @param neighborMetrics Map chứa LinkMetric của tất cả các node láng giềng.
     * @return Bảng định tuyến đã được tính toán.
     */
    RoutingTable computeRoutes(NodeInfo currentNode, Map<String, LinkMetric> neighborMetrics); // Đã sửa kiểu dữ liệu cho neighborMetrics

    /**
     * Tra cứu node kế tiếp (Next Hop) dựa trên điểm đến và bảng định tuyến hiện có.
     * @param packet Gói tin cần định tuyến (chứa Destination User ID).
     * @param routingTable Bảng định tuyến hiện tại của node.
     * @return ID của Node kế tiếp.
     */
    String getNextHop(Packet packet, RoutingTable routingTable);
}
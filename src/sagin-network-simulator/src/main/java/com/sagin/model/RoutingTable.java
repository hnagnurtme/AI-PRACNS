package com.sagin.model;

import lombok.*;
import java.util.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class RoutingTable {

    // Ánh xạ: destinationNodeId -> nextHopNodeId
    private Map<String, String> nextHopMap = new HashMap<>(); 

    // Ánh xạ: destinationNodeId -> toàn bộ path NodeId
    private Map<String, List<String>> fullPathMap = new HashMap<>();

    /**
     * Cập nhật toàn bộ bảng định tuyến bằng các Map mới.
     * Phương thức này được gọi sau khi RoutingEngine (ví dụ: Dijkstra) hoàn tất tính toán.
     * @param newNextHopMap Map next hop mới được tính toán.
     * @param newFullPathMap Map đường đi đầy đủ mới.
     */
    public void updateRoute(Map<String, String> newNextHopMap, Map<String, List<String>> newFullPathMap) {
        if (newNextHopMap != null) {
            // Sử dụng new HashMap() để tạo bản sao sâu, tránh tham chiếu trực tiếp
            this.nextHopMap = new HashMap<>(newNextHopMap);
        } else {
            this.nextHopMap = new HashMap<>();
        }
        
        if (newFullPathMap != null) {
            this.fullPathMap = new HashMap<>(newFullPathMap);
        } else {
            this.fullPathMap = new HashMap<>();
        }
    }
    
    /**
     * Cập nhật một tuyến đường cụ thể (destination -> nextHop).
     * Hữu ích cho các thuật toán định tuyến phản ứng (Reactive Routing) hoặc cập nhật láng giềng.
     */
    public void updateSingleRoute(String destinationNodeId, String nextHopNodeId, List<String> path) {
        nextHopMap.put(destinationNodeId, nextHopNodeId);
        fullPathMap.put(destinationNodeId, path != null ? new ArrayList<>(path) : new ArrayList<>());
    }


    /**
     * Lấy next hop dựa trên destination.
     * @param destinationNodeId ID đích cuối cùng.
     * @return ID của Node kế tiếp.
     */
    public String getNextHop(String destinationNodeId) {
        return nextHopMap.get(destinationNodeId);
    }

    /**
     * Lấy toàn bộ path history đến destination.
     * @param destinationNodeId ID đích cuối cùng.
     * @return Danh sách các Node trên đường đi.
     */
    public List<String> getPath(String destinationNodeId) {
        return fullPathMap.getOrDefault(destinationNodeId, new ArrayList<>());
    }
}
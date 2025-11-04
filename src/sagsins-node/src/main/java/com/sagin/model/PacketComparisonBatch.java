package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.types.ObjectId;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Model đại diện cho một batch gồm nhiều cặp packet (Dijkstra + RL)
 * Dùng để test mô phỏng với nhiều packets cùng lúc
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PacketComparisonBatch {
    
    @BsonId
    @JsonProperty("_id")
    private ObjectId id;
    
    /**
     * ID của batch, format: "batch_timestamp_random"
     */
    @JsonProperty("batchId")
    private String batchId;
    
    /**
     * Tổng số cặp packet trong batch này
     */
    @JsonProperty("totalPairPackets")
    private int totalPairPackets;
    
    /**
     * Số cặp đã hoàn thành (có đủ cả Dijkstra và RL packet)
     */
    @JsonProperty("completedPairs")
    private int completedPairs;
    
    /**
     * Danh sách các PacketComparison trong batch
     */
    @JsonProperty("packets")
    private List<PacketComparison> packets = new ArrayList<>();
    
    /**
     * Thời gian tạo batch
     */
    @JsonProperty("createdAt")
    private Instant createdAt;
    
    /**
     * Thời gian batch hoàn thành (khi tất cả packets đều complete)
     */
    @JsonProperty("completedAt")
    private Instant completedAt;
    
    /**
     * Trạng thái batch: "pending", "in_progress", "completed"
     */
    @JsonProperty("status")
    private String status;
    
    /**
     * Metadata bổ sung (VD: test scenario, người chạy test, etc.)
     */
    @JsonProperty("metadata")
    private BatchMetadata metadata;
    
    /**
     * Thống kê tổng hợp của batch
     */
    @JsonProperty("batchStatistics")
    private BatchStatistics batchStatistics;
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BatchMetadata {
        @JsonProperty("testScenario")
        private String testScenario;
        
        @JsonProperty("sourceUserId")
        private String sourceUserId;
        
        @JsonProperty("destinationUserId")
        private String destinationUserId;
        
        @JsonProperty("description")
        private String description;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BatchStatistics {
        // Dijkstra statistics
        @JsonProperty("dijkstra")
        private AlgorithmStats dijkstra;
        
        // RL statistics
        @JsonProperty("rl")
        private AlgorithmStats rl;
        
        // Comparison
        @JsonProperty("comparison")
        private ComparisonStats comparison;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AlgorithmStats {
        @JsonProperty("avgTotalLatency")
        private double avgTotalLatency;
        
        @JsonProperty("avgHopCount")
        private double avgHopCount;
        
        @JsonProperty("avgTotalDistance")
        private double avgTotalDistance;
        
        @JsonProperty("successCount")
        private int successCount;
        
        @JsonProperty("failureCount")
        private int failureCount;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ComparisonStats {
        @JsonProperty("dijkstraFasterCount")
        private int dijkstraFasterCount;
        
        @JsonProperty("rlFasterCount")
        private int rlFasterCount;
        
        @JsonProperty("dijkstraShorterCount")
        private int dijkstraShorterCount;
        
        @JsonProperty("rlShorterCount")
        private int rlShorterCount;
        
        @JsonProperty("avgLatencyDifference")
        private double avgLatencyDifference;
        
        @JsonProperty("avgDistanceDifference")
        private double avgDistanceDifference;
    }
}

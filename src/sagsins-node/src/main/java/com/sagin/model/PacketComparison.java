package com.sagin.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bson.BsonType;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.codecs.pojo.annotations.BsonProperty;
import org.bson.codecs.pojo.annotations.BsonRepresentation;

import java.time.Instant;

/**
 * Model để lưu kết quả so sánh giữa 2 packet (Dijkstra vs RL).
 * Cả 2 packet sẽ có cùng sourceUserId, destinationUserId để so sánh hiệu suất.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class PacketComparison {
    
    @BsonId
    @BsonRepresentation(BsonType.OBJECT_ID)
    private String id;
    
    /**
     * ID chung để nhóm 2 packet cùng 1 cặp so sánh
     */
    @BsonProperty("comparisonId")
    private String comparisonId;
    
    /**
     * Packet sử dụng thuật toán Dijkstra
     */
    @BsonProperty("dijkstraPacket")
    private Packet dijkstraPacket;
    
    /**
     * Packet sử dụng thuật toán RL (Reinforcement Learning)
     */
    @BsonProperty("rlPacket")
    private Packet rlPacket;
    
    /**
     * Thời điểm tạo bản ghi
     */
    @BsonProperty("createdAt")
    private Instant createdAt;
    
    /**
     * Trạng thái: "partial" (chỉ có 1 packet), "complete" (đủ 2 packet)
     */
    @BsonProperty("status")
    private String status;
    
    /**
     * Source và Destination để dễ query
     */
    @BsonProperty("sourceUserId")
    private String sourceUserId;
    
    @BsonProperty("destinationUserId")
    private String destinationUserId;
}

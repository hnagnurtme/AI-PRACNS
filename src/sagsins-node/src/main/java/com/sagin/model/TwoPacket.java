package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.types.ObjectId;

/**
 * Model đại diện cho 1 CẶP gói tin (Dijkstra + RL)
 * Lưu trong collection: two_packets
 * ✅ Mỗi cặp user CHỈ có 1 document (xóa và ghi đè mỗi lần)
 * ✅ PairId format: sourceUserId_destinationUserId
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class TwoPacket {
    
    @BsonId
    @JsonProperty("_id")
    private ObjectId id;
    
    /**
     * ID của cặp packet
     * ✅ Format: "sourceUserId_destinationUserId"
     * ✅ Mỗi cặp user CHỈ có 1 document
     */
    @JsonProperty("pairId")
    private String pairId;
    
    /**
     * Packet sử dụng thuật toán Dijkstra
     */
    @JsonProperty("dijkstraPacket")
    private Packet dijkstraPacket;
    
    /**
     * Packet sử dụng thuật toán RL
     */
    @JsonProperty("rlPacket")
    private Packet rlPacket;
}

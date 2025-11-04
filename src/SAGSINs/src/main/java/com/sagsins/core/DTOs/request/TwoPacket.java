package com.sagsins.core.DTOs.request;

import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.types.ObjectId;
import org.springframework.data.mongodb.core.mapping.Document;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.sagsins.core.model.Packet;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Document(collection = "two_packets")
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

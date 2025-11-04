package com.sagsins.core.DTOs.request;

import java.util.ArrayList;
import java.util.List;

import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.types.ObjectId;
import org.springframework.data.mongodb.core.mapping.Document;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Document(collection = "batch_packets")
public class BatchPacket {
    @BsonId
    @JsonProperty("_id")
    private ObjectId id;
    
    /**
     * ID của batch
     * ✅ Format: "sourceUserId_destinationUserId"
     */
    @JsonProperty("batchId")
    private String batchId;
    
    /**
     * Tổng số cặp packet trong batch
     */
    @JsonProperty("totalPairPackets")
    private int totalPairPackets;
    
    /**
     * Danh sách các cặp packets (TwoPacket)
     * ✅ Embedded documents
     */
    @JsonProperty("packets")
    private List<TwoPacket> packets = new ArrayList<>();
}

package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.types.ObjectId;

import java.util.ArrayList;
import java.util.List;

/**
 * Model đại diện cho 1 LÔ gói tin chứa nhiều CẶP (Dijkstra + RL)
 * Lưu trong collection: batch_packets
 * ✅ BatchId format: sourceUserId_destinationUserId
 * ✅ Nếu trùng batchId → xóa document cũ, tạo mới
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
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

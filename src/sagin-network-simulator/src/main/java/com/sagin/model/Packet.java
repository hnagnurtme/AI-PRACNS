package com.sagin.model;

import java.io.Serializable;
import java.util.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@ToString
public class Packet implements Serializable {
    
    // --- ENUM: Loại Gói tin (Dùng cho ACK) ---
    public enum PacketType {
        DATA,  // Gói tin dữ liệu thông thường
        ACK    // Gói tin Xác nhận
    }
    
    // --- ID và Địa chỉ ---
    private String packetId;           
    private String sourceUserId;       
    private String destinationUserId;  
    
    // --- Trạng thái Thời gian và Phân loại ---
    private PacketType type = PacketType.DATA; // Mặc định là DATA
    private String acknowledgedPacketId;       // ID của gói tin DATA mà ACK này xác nhận
    private long timeSentFromSourceMs;         // Thời điểm gửi từ nguồn ban đầu

    // --- Dữ liệu Ứng dụng ---
    private String payloadDataBase64;   
    private int payloadSizeByte;        
    private String serviceType;         

    // --- Định tuyến và Theo dõi ---
    private int TTL;                   
    private String currentHoldingNodeId; 
    private String nextHopNodeId;      
    private List<String> pathHistory;  
    private double accumulatedDelayMs;  
    private int priorityLevel;          

    // --- QoS Yêu cầu (Được lấy từ ServiceQoS) ---
    private double maxAcceptableLatencyMs; 
    private double maxAcceptableLossRate;  
    
    // --- Trạng thái Drop ---
    private boolean dropped = false;           
    private String dropReason;          

    /** Thêm node hiện tại vào lịch sử đường đi. */
    public void addToPath(String nodeId) {
        if (pathHistory == null) {
            // Sử dụng LinkedList nếu bạn cần chèn/xóa nhanh, nhưng ArrayList là đủ cho lịch sử.
            pathHistory = new ArrayList<>();
        }
        pathHistory.add(nodeId);
    }

    /** Giảm Time-To-Live. */
    public void decrementTTL() {
        if (TTL > 0) {
            TTL--;
        }
    }

    /** Kiểm tra gói tin còn khả dụng (chưa bị drop và TTL > 0). */
    public boolean isAlive() {
        return TTL > 0 && !dropped;
    }

    /** Đánh dấu gói tin bị hủy. */
    public void markDropped(String reason) {
        this.dropped = true;
        this.dropReason = reason;
    }
}
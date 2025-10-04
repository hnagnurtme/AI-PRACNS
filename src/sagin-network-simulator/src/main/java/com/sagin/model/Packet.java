package com.sagin.model;

import java.util.ArrayList;
import java.util.List;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public class Packet {
    
    // Thuộc tính giữ nguyên như cũ...
    private String packetId;           
    private String sourceUserId;       
    private String destinationUserId;  
    private long timestamp;            

    private String payloadDataBase64;   
    private int payloadSizeByte;        
    private String serviceType;         

    private int TTL;                   
    private String currentHoldingNodeId; 
    private String nextHopNodeId;      
    private List<String> pathHistory;  

    private double accumulatedDelayMs;  
    private int priorityLevel;          
    private double maxAcceptableLatencyMs; 
    private double maxAcceptableLossRate;  
    
    private boolean dropped;           

    /** Thêm node hiện tại vào lịch sử đường đi. */
    public void addToPath(String nodeId) {
        if (pathHistory == null) {
            pathHistory = new ArrayList<>();
        }
        pathHistory.add(nodeId);
    }

    /** Giảm Time-To-Live. */
    public void decrementTTL() {
        TTL--;
    }

    /** Kiểm tra gói tin còn khả dụng (chưa bị drop và TTL > 0). */
    public boolean isAlive() {
        return TTL > 0 && !dropped;
    }

    /** Đánh dấu gói tin bị hủy. */
    public void markDropped() {
        this.dropped = true;
    }
}
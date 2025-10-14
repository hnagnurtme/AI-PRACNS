package com.sagin.model;

import java.io.Serializable;
import java.util.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

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
    @JsonProperty("TTL")
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

    // getter và setter
    public String getPacketId() {
        return packetId;
    }
    public void setPacketId(String packetId) {
        this.packetId = packetId;
    }
    public String getSourceUserId() {
        return sourceUserId;
    }   
    public void setSourceUserId(String sourceUserId) {
        this.sourceUserId = sourceUserId;
    }
    public String getDestinationUserId() {
        return destinationUserId;
    }
    public void setDestinationUserId(String destinationUserId) {
        this.destinationUserId = destinationUserId;
    }
    public PacketType getType() {
        return type;
    }   
    public void setType(PacketType type) {
        this.type = type;
    }
    public String getAcknowledgedPacketId() {
        return acknowledgedPacketId;
    }
    public void setAcknowledgedPacketId(String acknowledgedPacketId) {
        this.acknowledgedPacketId = acknowledgedPacketId;
    }   
    public long getTimeSentFromSourceMs() {
        return timeSentFromSourceMs;
    }
    public void setTimeSentFromSourceMs(long timeSentFromSourceMs) {
        this.timeSentFromSourceMs = timeSentFromSourceMs;
    }   
    public String getPayloadDataBase64() {
        return payloadDataBase64;
    }   
    public void setPayloadDataBase64(String payloadDataBase64) {
        this.payloadDataBase64 = payloadDataBase64;
    }
    public int getPayloadSizeByte() {
        return payloadSizeByte;
    }
    public void setPayloadSizeByte(int payloadSizeByte) {
        this.payloadSizeByte = payloadSizeByte;
    }
    public String getServiceType() {
        return serviceType;
    }   
    public void setServiceType(String serviceType) {
        this.serviceType = serviceType;
    }
    public int getTTL() {
        return TTL;
    }
    public void setTTL(int tTL) {
        TTL = tTL;
    }
    public String getCurrentHoldingNodeId() {
        return currentHoldingNodeId;
    }
    public void setCurrentHoldingNodeId(String currentHoldingNodeId) {
        this.currentHoldingNodeId = currentHoldingNodeId;
    }
    public String getNextHopNodeId() {
        return nextHopNodeId;
    }
    public void setNextHopNodeId(String nextHopNodeId) {
        this.nextHopNodeId = nextHopNodeId;
    }
    public List<String> getPathHistory() {
        return pathHistory;
    }
    public void setPathHistory(List<String> pathHistory) {
        this.pathHistory = pathHistory;
    }
    public double getAccumulatedDelayMs() {
        return accumulatedDelayMs;
    }
    public void setAccumulatedDelayMs(double accumulatedDelayMs) {
        this.accumulatedDelayMs = accumulatedDelayMs;
    }
    public int getPriorityLevel() {
        return priorityLevel;
    }
    public void setPriorityLevel(int priorityLevel) {
        this.priorityLevel = priorityLevel;
    }
    public double getMaxAcceptableLatencyMs() {
        return maxAcceptableLatencyMs;

    }
    public void setMaxAcceptableLatencyMs(double maxAcceptableLatencyMs) {
        this.maxAcceptableLatencyMs = maxAcceptableLatencyMs;
    }
    public double getMaxAcceptableLossRate() {
        return maxAcceptableLossRate;
    }
    public void setMaxAcceptableLossRate(double maxAcceptableLossRate) {
        this.maxAcceptableLossRate = maxAcceptableLossRate;
    }
    public boolean isDropped() {
        return dropped;
    }
    public void setDropped(boolean dropped) {
        this.dropped = dropped;
    }
    public String getDropReason() {
        return dropReason;
    }
    public void setDropReason(String dropReason) {
        this.dropReason = dropReason;
    }
}
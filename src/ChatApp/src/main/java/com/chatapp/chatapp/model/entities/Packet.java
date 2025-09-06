package com.chatapp.chatapp.model.entities;

import java.util.ArrayList;
import java.util.List;

public class Packet {
    
    // ----- Identification -----
    private String packetId;            // ID duy nhất của packet
    private String sourceUserId;        // User gửi
    private String destinationUserId;   // User nhận
    private long timestamp;             // Thời điểm tạo packet (millis)

    // ----- Payload -----
    private String message;             // Nội dung tin nhắn
    private int payloadSize;            // Kích thước dữ liệu

    // ----- Routing / Forwarding -----
    private int TTL;                    // Time-To-Live
    private String currentNode;         // Node đang giữ packet
    private String nextHop;             // Node sẽ gửi tiếp
    private List<String> pathHistory;   // Lịch sử các node đã đi qua

    // ----- QoS / Metrics -----
    private double delayMs;             // Tổng delay từ nguồn
    private double lossRate;            // Tỷ lệ mất packet
    private int retryCount;             // Số lần retry nếu link chưa sẵn sàng
    private int priority;               // Mức ưu tiên (1 cao nhất)

    // ----- Status -----
    private boolean dropped;            // Packet đã bị drop chưa

    // Constructor
    public Packet() {
        this.timestamp = System.currentTimeMillis();
        this.pathHistory = new ArrayList<>();
        this.TTL = 64;
        this.priority = 5;
        this.retryCount = 0;
        this.dropped = false;
        this.delayMs = 0.0;
        this.lossRate = 0.0;
    }

    public Packet(String sourceUserId, String destinationUserId, String message) {
        this();
        this.packetId = generatePacketId();
        this.sourceUserId = sourceUserId;
        this.destinationUserId = destinationUserId;
        this.message = message;
        this.payloadSize = message != null ? message.length() : 0;
    }

    // Factory methods
    public static Packet createMessage(String from, String to, String message) {
        return new Packet(from, to, message);
    }

    public static Packet createBroadcast(String from, String message) {
        return new Packet(from, "BROADCAST", message);
    }

    // Generate unique packet ID
    private String generatePacketId() {
        return "PKT_" + System.currentTimeMillis() + "_" + (int)(Math.random() * 1000);
    }

    // Getters and Setters
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

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
        this.payloadSize = message != null ? message.length() : 0;
    }

    public int getPayloadSize() {
        return payloadSize;
    }

    public void setPayloadSize(int payloadSize) {
        this.payloadSize = payloadSize;
    }

    public int getTTL() {
        return TTL;
    }

    public void setTTL(int TTL) {
        this.TTL = TTL;
    }

    public String getCurrentNode() {
        return currentNode;
    }

    public void setCurrentNode(String currentNode) {
        this.currentNode = currentNode;
    }

    public String getNextHop() {
        return nextHop;
    }

    public void setNextHop(String nextHop) {
        this.nextHop = nextHop;
    }

    public List<String> getPathHistory() {
        return pathHistory;
    }

    public void setPathHistory(List<String> pathHistory) {
        this.pathHistory = pathHistory;
    }

    public double getDelayMs() {
        return delayMs;
    }

    public void setDelayMs(double delayMs) {
        this.delayMs = delayMs;
    }

    public double getLossRate() {
        return lossRate;
    }

    public void setLossRate(double lossRate) {
        this.lossRate = lossRate;
    }

    public int getRetryCount() {
        return retryCount;
    }

    public void setRetryCount(int retryCount) {
        this.retryCount = retryCount;
    }

    public int getPriority() {
        return priority;
    }

    public void setPriority(int priority) {
        this.priority = priority;
    }

    public boolean isDropped() {
        return dropped;
    }

    public void setDropped(boolean dropped) {
        this.dropped = dropped;
    }

    @Override
    public String toString() {
        return String.format("Packet{id='%s', from='%s', to='%s', message='%s', dropped=%s}", 
            packetId, sourceUserId, destinationUserId, message, dropped);
    }
}
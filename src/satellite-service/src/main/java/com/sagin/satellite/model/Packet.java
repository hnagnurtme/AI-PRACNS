package com.sagin.satellite.model;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class Packet {
    // ----- Identification -----
    private String packetId;           // ID duy nhất của packet
    private String sourceUserId;       // User gửi
    private String destinationUserId;  // User nhận
    private long timestamp;            // Thời điểm tạo packet (millis)

    // ----- Payload -----
    private String message;            // Nội dung tin nhắn
    private int payloadSize;           // Kích thước dữ liệu

    // ----- Routing / Forwarding -----
    private int TTL;                   // Time-To-Live
    private String currentNode;        // Node đang giữ packet
    private String nextHop;            // Node sẽ gửi tiếp
    private List<String> pathHistory;  // Lịch sử các node đã đi qua

    // ----- QoS / Metrics -----
    private double delayMs;            // Tổng delay từ nguồn
    private double lossRate;           // Tỷ lệ mất packet
    private int retryCount;            // Số lần retry nếu link chưa sẵn sàng
    private int priority;              // Mức ưu tiên (1 cao nhất)
    
    // ----- Status -----
    private boolean dropped;           // Packet đã bị drop chưa


    public void addToPath(String nodeId) {
        if (pathHistory == null) {
            pathHistory = new ArrayList<>();
        }
        pathHistory.add(nodeId);
    }

    public void incrementRetry() {
        retryCount++;
    }

    public void markDropped() {
        dropped = true;
    }

    public boolean isAlive() {
        return TTL > 0 && !dropped;
    }
}

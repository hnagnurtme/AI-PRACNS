package com.chatapp.chatapp.model.interfaces;

import java.time.LocalDateTime;
import java.util.List;

public interface LogMonitoringService {
    void logPacket(PacketLog packet);
    List<PacketLog> getPacketHistory();
    List<PacketLog> filterPackets(String filter);
    void exportLog(String format);
    PerformanceMetrics getCurrentMetrics();

    record PacketLog(String packetId, String source, String destination, String status, long delay, LocalDateTime timestamp) {}
    record PerformanceMetrics(double throughput, long latency, double packetLoss) {}
}

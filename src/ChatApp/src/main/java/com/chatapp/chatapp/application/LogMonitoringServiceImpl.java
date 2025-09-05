package com.chatapp.chatapp.application;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import com.chatapp.chatapp.gui.interfaces.LogMonitoringService;

import javafx.scene.control.ComboBox;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;

public class LogMonitoringServiceImpl implements LogMonitoringService {
    private final TableView<Object> packetLogTable; 
    private final TextField searchField;
    private final ComboBox<String> filterCombo;
    private final List<PacketLog> logs = new ArrayList<>();
    
    public LogMonitoringServiceImpl(TableView<Object> packetLogTable, TextField searchField, ComboBox<String> filterCombo) {
        this.packetLogTable = packetLogTable;
        this.searchField = searchField;
        this.filterCombo = filterCombo;
        
        setupLogPanel();
        initializeMockData();
    }
    
    private void setupLogPanel() {
        if (searchField != null) {
            searchField.setPromptText("Search packets...");
        }
        
        if (filterCombo != null) {
            filterCombo.getItems().addAll("All", "Sent", "Received", "Failed");
            filterCombo.setValue("All");
        }
    }
    
    @Override
    public void logPacket(PacketLog packet) {
        logs.add(packet);
        System.out.println("Logged packet: " + packet.packetId());
    }
    
    @Override
    public List<PacketLog> getPacketHistory() {
        return new ArrayList<>(logs);
    }
    
    @Override
    public List<PacketLog> filterPackets(String filter) {
        if (filter == null || filter.equals("All")) {
            return getPacketHistory();
        }
        
        return logs.stream()
            .filter(log -> log.status().equalsIgnoreCase(filter) ||
                          log.source().contains(filter) ||
                          log.destination().contains(filter))
            .toList();
    }
    
    @Override
    public void exportLog(String format) {
        System.out.println("Exporting " + logs.size() + " logs in " + format + " format");
    }
    
    @Override
    public PerformanceMetrics getCurrentMetrics() {
        double throughput = logs.size() * 1.5;
        long avgLatency = logs.isEmpty() ? 0 : logs.stream().mapToLong(PacketLog::delay).sum() / logs.size() / 1000000;
        double packetLoss = logs.stream().mapToDouble(log -> "failed".equals(log.status()) ? 1.0 : 0.0).average().orElse(0.0);
        
        return new PerformanceMetrics(throughput, avgLatency, packetLoss);
    }
    
    public void exportLogAction() {
        exportLog("CSV");
    }
    
    private void initializeMockData() {
        LocalDateTime now = LocalDateTime.now();
        logs.add(new PacketLog("PKT001", "Starlink", "HaNoi", "sent", 120000000L, now.minusMinutes(5)));
        logs.add(new PacketLog("PKT002", "ERS-1", "Adelaide", "received", 180000000L, now.minusMinutes(3)));
        logs.add(new PacketLog("PKT003", "Vinasat 1", "Cape Canaveral", "failed", 0L, now.minusMinutes(1)));
    }
}
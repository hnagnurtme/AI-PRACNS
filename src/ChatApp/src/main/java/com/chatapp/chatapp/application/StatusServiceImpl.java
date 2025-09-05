package com.chatapp.chatapp.application;

import com.chatapp.chatapp.gui.interfaces.StatusService;

import javafx.scene.control.Label;

public class StatusServiceImpl implements StatusService {
    private final Label connectionStatusLabel;
    private final Label currentNodeLabel;
    private final Label networkStatsLabel;
    
    public StatusServiceImpl(Label connectionStatusLabel, Label currentNodeLabel, Label networkStatsLabel) {
        this.connectionStatusLabel = connectionStatusLabel;
        this.currentNodeLabel = currentNodeLabel;
        this.networkStatsLabel = networkStatsLabel;
        
        updateConnectionStatus("Disconnected");
        updateCurrentNode("None");
        updateNetworkStats("0 packets/s");
    }
    
    @Override
    public void updateConnectionStatus(String status) {
        if (connectionStatusLabel != null) {
            connectionStatusLabel.setText("Status: " + status);
        }
    }
    
    @Override
    public void updateCurrentNode(String node) {
        if (currentNodeLabel != null) {
            currentNodeLabel.setText("Node: " + node);
        }
    }
    
    @Override
    public void updateNetworkStats(String stats) {
        if (networkStatsLabel != null) {
            networkStatsLabel.setText("Network: " + stats);
        }
    }
}   
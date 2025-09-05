package com.chatapp.chatapp.application;

import java.util.List;

import com.chatapp.chatapp.gui.interfaces.AccessPointService;

import javafx.application.Platform;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;

public class AccessPointServiceImpl implements AccessPointService {
    private final ComboBox<String> accessPointCombo;
    private final Label aiRecommendationLabel;
    private final Label connectionIndicator;
    private ConnectionStatus currentStatus = ConnectionStatus.DISCONNECTED;
    
    public AccessPointServiceImpl(ComboBox<String> accessPointCombo, Label aiRecommendationLabel, Label connectionIndicator) {
        this.accessPointCombo = accessPointCombo;
        this.aiRecommendationLabel = aiRecommendationLabel;
        this.connectionIndicator = connectionIndicator;
        
        setupInitialAccessPoints();
    }
    
    private void setupInitialAccessPoints() {
        if (accessPointCombo != null) {
            accessPointCombo.getItems().addAll("Starlink", "ERS-1", "Vinasat 1", "Cape Canaveral", "Adelaide", "HaNoi");
            accessPointCombo.setValue("Starlink");
        }
    }
    
    @Override
    public List<AccessPoint> getAvailableAccessPoints() {
        return List.of(
            new AccessPoint("starlink", "Starlink", "satellite", 0.95, "Global"),
            new AccessPoint("ers-1", "ERS-1", "satellite", 0.85, "Europe"),
            new AccessPoint("vinasat1", "Vinasat 1", "satellite", 0.80, "Vietnam"),
            new AccessPoint("cape_canaveral", "Cape Canaveral", "ground", 0.98, "USA"),
            new AccessPoint("adelaide", "Adelaide", "ground", 0.92, "Australia"),
            new AccessPoint("hanoi", "HaNoi", "ground", 0.88, "Vietnam")
        );
    }
    
    @Override
    public void connectToAccessPoint(String accessPointId) {
        currentStatus = ConnectionStatus.CONNECTING;
        updateConnectionIndicator("Connecting to " + accessPointId + "...");
        
        new Thread(() -> {
            try {
                Thread.sleep(2000);
                Platform.runLater(() -> {
                    currentStatus = ConnectionStatus.CONNECTED;
                    updateConnectionIndicator("Connected");
                });
            } catch (Exception e) {
                e.printStackTrace();
                Platform.runLater(() -> {
                    currentStatus = ConnectionStatus.ERROR;
                    updateConnectionIndicator("Connection failed");
                });
            }
        }).start();
    }
    
    @Override
    public void disconnectFromAccessPoint() {
        currentStatus = ConnectionStatus.DISCONNECTED;
        updateConnectionIndicator("Disconnected");
    }
    
    @Override
    public AccessPoint getAIRecommendation() {
        // Logic từ Controller cũ
        if (aiRecommendationLabel != null) {
            aiRecommendationLabel.setText("AI Recommendation: ERS-1 (95% confidence)\nReason: Optimal signal strength and low latency");
        }
        
        if (accessPointCombo != null) {
            accessPointCombo.setValue("ERS-1");
        }
        
        return new AccessPoint("ers-1", "ERS-1", "satellite", 0.95, "Europe");
    }
    
    @Override
    public ConnectionStatus getConnectionStatus() {
        return currentStatus;
    }
    
    public void quickConnect() {
        if (accessPointCombo != null) {
            String selectedAccessPoint = accessPointCombo.getValue();
            if (selectedAccessPoint != null) {
                connectToAccessPoint(selectedAccessPoint);
            }
        }
    }
    
    private void updateConnectionIndicator(String status) {
        if (connectionIndicator != null) {
            connectionIndicator.setText("Status: " + status);
        }
    }
}
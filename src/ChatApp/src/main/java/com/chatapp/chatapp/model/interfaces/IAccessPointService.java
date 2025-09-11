package com.chatapp.chatapp.model.interfaces;

import java.util.List;

public interface IAccessPointService {
    List<AccessPoint> getAvailableAccessPoints();
    void connectToAccessPoint(String accessPointId);
    void disconnectFromAccessPoint();
    AccessPoint getAIRecommendation();
    ConnectionStatus getConnectionStatus();

    record AccessPoint(String id, String name, String type, double signalStrength, String location) {}
    enum ConnectionStatus { DISCONNECTED, CONNECTING, CONNECTED, ERROR }
}

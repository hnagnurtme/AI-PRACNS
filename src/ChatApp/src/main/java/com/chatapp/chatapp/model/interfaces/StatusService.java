package com.chatapp.chatapp.model.interfaces;

public interface StatusService {
    void updateConnectionStatus(String status);
    void updateCurrentNode(String node);
    void updateNetworkStats(String stats);
}

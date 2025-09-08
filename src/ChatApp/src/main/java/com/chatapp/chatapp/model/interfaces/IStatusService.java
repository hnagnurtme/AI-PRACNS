package com.chatapp.chatapp.model.interfaces;

public interface IStatusService {
    void updateConnectionStatus(String status);
    void updateCurrentNode(String node);
    void updateNetworkStats(String stats);
}

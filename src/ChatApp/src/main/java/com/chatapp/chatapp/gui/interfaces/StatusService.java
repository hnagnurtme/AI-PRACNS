package com.chatapp.chatapp.gui.interfaces;

public interface StatusService {
    void updateConnectionStatus(String status);
    void updateCurrentNode(String node);
    void updateNetworkStats(String stats);
}

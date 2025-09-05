package com.chatapp.chatapp.gui.interfaces;

public interface MainWindowView {
    ChatService getChatService();
    NetworkTopologyService getTopologyService();
    AccessPointService getAccessPointService();
    LogMonitoringService getLogService();
    ThemeService getThemeService();
    StatusService getStatusService();

    void initializeView();
    void shutdown();
}

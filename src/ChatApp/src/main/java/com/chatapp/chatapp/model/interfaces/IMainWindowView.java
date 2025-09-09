package com.chatapp.chatapp.model.interfaces;

public interface IMainWindowView {
    IChatService getChatService();
    INetworkTopologyService getTopologyService();
    IAccessPointService getAccessPointService();
    ILogMonitoringService getLogService();
    IThemeService getThemeService();
    IStatusService getStatusService();

    void initializeView();
    void shutdown();
}

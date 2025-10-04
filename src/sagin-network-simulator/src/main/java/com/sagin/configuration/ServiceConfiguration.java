package com.sagin.configuration;

import com.sagin.core.ILinkManagerService;
import com.sagin.core.INetworkManagerService;
import com.sagin.core.service.LinkManagerService;
import com.sagin.core.service.NetworkManagerService;
import com.sagin.routing.QosDijkstraEngine; 
import com.sagin.routing.RoutingEngine; 

/**
 * Lớp cấu hình chính, chịu trách nhiệm khởi tạo và cung cấp các phiên bản Singleton
 * của các dịch vụ cốt lõi trong hệ thống mô phỏng.
 */
public class ServiceConfiguration {

    // --- Singleton Instances ---

    private final ILinkManagerService linkManagerService;
    private final RoutingEngine routingEngine;
    private final INetworkManagerService networkManagerService;

    // --------------------------------------------------------------------------

    public ServiceConfiguration() {
        this.linkManagerService = new LinkManagerService();
        this.routingEngine = new QosDijkstraEngine();
        this.networkManagerService = new NetworkManagerService();
    }
    
    // --- Public Getters ---

    public ILinkManagerService getLinkManagerService() {
        return linkManagerService;
    }

    /**
     * Trả về cơ chế định tuyến (QoS Dijkstra Engine hoặc RL Agent sau này).
     * @return RoutingEngine
     */
    public RoutingEngine getRoutingEngine() {
        return routingEngine;
    }

    public INetworkManagerService getNetworkManagerService() {
        return networkManagerService;
    }
    
    // --------------------------------------------------------------------------

    /**
     * Khởi tạo một phiên bản Singleton của ServiceConfiguration.
     */
    private static class SingletonHelper {
        private static final ServiceConfiguration INSTANCE = new ServiceConfiguration();
    }

    /**
     * Phương thức tĩnh để lấy instance của ServiceConfiguration.
     */
    public static ServiceConfiguration getInstance() {
        return SingletonHelper.INSTANCE;
    }
}
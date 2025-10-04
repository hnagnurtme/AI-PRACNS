package com.sagin.configuration;

import com.sagin.core.ILinkManagerService;
import com.sagin.core.INetworkManagerService;
import com.sagin.core.service.LinkManagerService;
import com.sagin.core.service.NetworkManagerService;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.FirebaseNodeRepository;
import com.sagin.routing.QosDijkstraEngine; 
import com.sagin.routing.RoutingEngine; 

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Lớp cấu hình chính, chịu trách nhiệm khởi tạo và cung cấp các phiên bản Singleton
 * của các dịch vụ cốt lõi trong hệ thống mô phỏng.
 */
public class ServiceConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(ServiceConfiguration.class);

    // --- Singleton Instances ---
    private final ILinkManagerService linkManagerService;
    private final RoutingEngine routingEngine;
    private final INetworkManagerService networkManagerService;
    private final INodeRepository nodeRepository; 

    // --------------------------------------------------------------------------

    public ServiceConfiguration() {
        // 1. KHỞI TẠO FIREBASE CONNECTION VÀ REPOSITORY
        logger.info("Bắt đầu khởi tạo kết nối Firebase và các dịch vụ mạng...");

        // Khởi tạo kết nối tĩnh đến Firestore (Sẽ ném IOException nếu key sai/thiếu)
        try {
            FireStoreConfiguration.init();
        } catch (IOException e) {
            logger.error("LỖI NGHIÊM TRỌNG: Không thể khởi tạo Firestore. Hãy kiểm tra tệp {}.", 
                         FirebaseConfiguration.SERVICE_ACCOUNT_FILE);
            logger.error("Chi tiết lỗi:", e);
            // Trong môi trường Docker, việc này có thể yêu cầu dừng ứng dụng
            throw new RuntimeException("Lỗi cấu hình Firebase, ứng dụng dừng lại.", e); 
        }

        this.nodeRepository = new FirebaseNodeRepository();
        this.linkManagerService = new LinkManagerService();
        this.routingEngine = new QosDijkstraEngine();
        
        // 2. DEPENDENCY INJECTION: NetworkManagerService nhận Repository
        this.networkManagerService = new NetworkManagerService(this.nodeRepository);

        logger.info("Tất cả các dịch vụ mạng đã được cấu hình và kết nối thành công.");
    }
    
    // --- Public Getters ---

    public ILinkManagerService getLinkManagerService() {
        return linkManagerService;
    }

    public RoutingEngine getRoutingEngine() {
        return routingEngine;
    }

    public INetworkManagerService getNetworkManagerService() {
        return networkManagerService;
    }

    public INodeRepository getNodeRepository() {
        return nodeRepository;
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
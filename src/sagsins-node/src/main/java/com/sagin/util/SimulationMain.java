package com.sagin.util;
import com.sagin.model.NodeInfo;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.MongoNodeRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);
    private static final int FALLBACK_PORT = 8080; 

    public static void main(String[] args) {
        String nodeId;

        // --- LOGIC MẶC ĐỊNH VÀ XỬ LÝ THAM SỐ ---
        if (args.length < 1) { // Chỉ kiểm tra 1 tham số
            nodeId = "GS-01";
            logger.warn("Thiếu tham số khởi chạy. Sử dụng mặc định: NODE_ID={}", nodeId);
        } else {
            nodeId = args[0];
        }
        // ----------------------------------------

        logger.info("--- KHỞI CHẠY NODE ĐỘC LẬP: {} ---", nodeId);
        new SimulationMain().runSingleNode(nodeId); // Chỉ truyền NODEID
    }
    
    public void runSingleNode(String nodeId) {

        // --- 1. KHỞI TẠO REPOSITORY VÀ SEEDING DỮ LIỆU ---
        INodeRepository nodeRepository = new MongoNodeRepository();
        
        // Tải cấu hình Node CỤ THỂ
        NodeInfo initialInfo = nodeRepository
            .getNodeInfo(nodeId)
            .orElseThrow(() -> new IllegalArgumentException(
                "KHÔNG TÌM THẤY CẤU HÌNH NODE TRONG DB CHO NODE_ID=" + nodeId
            ));
        logger.info("Cau hinh : {}", initialInfo.toString());

        // --- 1B. XÁC ĐỊNH CỔNG VÀ CẬP NHẬT ĐỊA CHỈ MẠNG CỤC BỘ ---
        // Port được lấy từ DB (nếu có) hoặc dùng mặc định.
        int port = initialInfo.getPort() > 0 ? initialInfo.getPort() : FALLBACK_PORT;
        
        try {
            String hostName = "localhost";
            
            logger.info("Địa chỉ mạng hiện tại của Node {}: {}:{}", nodeId, hostName, port);
            // Set host/port in-memory only. We intentionally do NOT persist the host
            // (or overwrite the DB copy) here to avoid seeding/runtime logic that
            // would set the host equal to the nodeId or otherwise leak local values
            // into the shared database.
            initialInfo.setHost(hostName);
            initialInfo.setPort(port);

            // NOTE: Do NOT call nodeRepository.updateNodeInfo(...) here. Persisting
            // the host during node startup can overwrite the intended DB-stored
            // network configuration (and previously was set from nodeId). Keep the
            // change local to this process only.
            logger.info("Địa chỉ mạng Node {} đã được cập nhật cục bộ: {}:{} (không ghi lên DB)", nodeId, hostName, port);
            
        } catch (Exception e) {
            logger.error("LỖI CẬP NHẬT ĐỊA CHỈ MẠNG: Không thể xác định Hostname. {}", e.getMessage());
            return;
        }
        
    }
}
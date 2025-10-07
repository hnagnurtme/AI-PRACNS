package com.sagin.core.service;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.repository.INodeRepository;
import com.sagin.util.PacketSerializerHelper;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lớp triển khai INetworkManagerService.
 * Quản lý và điều phối tất cả NodeService đang chạy.
 * Lớp này hoạt động như một Registry và API Gateway cho mạng mô phỏng.
 */
public class NetworkManagerService implements INetworkManagerService {

    private static final Logger logger = LoggerFactory.getLogger(NetworkManagerService.class);

    // Lưu trữ tất cả NodeService đang hoạt động (Node ID -> NodeService Object)
    private final Map<String, INodeService> activeNodeServices;
    // Lưu trữ tất cả NodeInfo (Database Vị trí trong bộ nhớ)
    private final Map<String, NodeInfo> networkNodesInfo;

    // DEPENDENCY: Repository để tải dữ liệu từ DB
    private final INodeRepository nodeRepository;

    public NetworkManagerService(INodeRepository nodeRepository) { 
        this.activeNodeServices = new ConcurrentHashMap<>();
        this.networkNodesInfo = new ConcurrentHashMap<>();
        this.nodeRepository = nodeRepository;
        logger.info("NetworkManagerService đã khởi tạo.");
    }

    @Override
    public void initializeNetwork(Map<String, NodeInfo> initialNodeConfigs) {
        logger.info("Khởi tạo cấu trúc mạng: Bắt đầu tải dữ liệu Node...");

        Map<String, NodeInfo> dbConfigs = nodeRepository.loadAllNodeConfigs();

        this.networkNodesInfo.putAll(dbConfigs);

        this.networkNodesInfo.putAll(initialNodeConfigs);

        logger.info("Tải thành công {} Node (Bao gồm cả Node đang chạy) vào Registry.",
                this.networkNodesInfo.size());
    }

    @Override
    public void registerActiveNode(String serviceId, INodeService nodeService) {
        if (!activeNodeServices.containsKey(serviceId)) {
            activeNodeServices.put(serviceId, nodeService);
            logger.info("Node {} đã đăng ký thành công vào NetworkManager.", serviceId);
        } else {
            logger.warn("Node {} đã tồn tại trong danh sách Node hoạt động (Đã đăng ký lại).", serviceId);
        }
    }

    @Override
    public void transferPacket(Packet packet, String destNodeId) {
        // 1. LẤY ĐỊA CHỈ IP và PORT CỦA NODE ĐÍCH
        String destAddress = "UAV_001:4001";

        // if (destAddress == null) {
        //     logger.warn("LỖI CHUYỂN GIAO: Không tìm thấy địa chỉ IP/Port cho Node đích {}.", destNodeId);
        //     packet.markDropped();
        //     return;
        // }

        // Phân tích IP và Port
        String[] parts = destAddress.split(":");
        if (parts.length != 2) {
            logger.error("Địa chỉ Node đích không hợp lệ: {}", destAddress);
            packet.markDropped();
            return;
        }
        String destIp = parts[0];
        int destPort = Integer.parseInt(parts[1]);

        // 2. TUẦN TỰ HÓA (SERIALIZE) GÓI TIN SANG JSON
        String jsonPayload = PacketSerializerHelper.serialize(packet);
        if (jsonPayload == null) {
            logger.error("LỖI CHUYỂN GIAO: Không thể tuần tự hóa gói tin {}.", packet.getPacketId());
            packet.markDropped();
            return;
        }
        logger.info("Packet {} đã được tuần tự hóa thành công.", packet.getPacketId());
        logger.info(destIp + ":" + destPort);

        // 3. THIẾT LẬP VÀ GỬI QUA TCP SOCKET
        try (
                // Mở Socket Client và kết nối đến Node đích
                Socket socket = new Socket(destIp, destPort);

                // Dùng PrintWriter để gửi dữ liệu dạng chuỗi
                PrintWriter out = new PrintWriter(socket.getOutputStream(), true); // 'true' để autoFlush
        ) {
            logger.info("Chuyển giao: Gói {} từ {} -> {} qua TCP {}:{}",
                    packet.getPacketId(), packet.getCurrentHoldingNodeId(), destNodeId, destIp, destPort);

            // Gửi chuỗi JSON. out.println() sẽ tự động thêm ký tự xuống dòng (\n)
            // giúp server đích dễ dàng đọc theo từng dòng tin nhắn.
            out.println(jsonPayload);

            // 4. CẬP NHẬT TRẠNG THÁI NỘI BỘ (nếu cần)
            // Logic cập nhật lên DB (Firebase) giữ nguyên: cập nhật thông tin Node hiện tại
            NodeInfo currentNodeInfo = networkNodesInfo.get(packet.getCurrentHoldingNodeId());
            if (currentNodeInfo != null) {
                // Cập nhật trạng thái (ví dụ: giảm tải/hàng đợi) trên Node hiện tại sau khi gửi
                // thành công
                nodeRepository.updateNodeInfo(packet.getCurrentHoldingNodeId(), currentNodeInfo);
            }

            logger.info("Gói tin {} đã được gửi qua TCP thành công.", packet.getPacketId());

        } catch (IOException e) {
            // Xử lý các lỗi kết nối mạng (Network failure)
            logger.error("LỖI CHUYỂN GIAO MẠNG (TCP): Không thể kết nối hoặc gửi gói tin đến {}:{}. Lỗi: {}",
                    destIp, destPort, e.getMessage());
            packet.markDropped();
        }
    }

    @Override
    public NodeInfo getNodeInfo(String nodeId) {
        // Cung cấp thông tin của các node khác trong mạng (cho khám phá láng giềng)
        return networkNodesInfo.get(nodeId);
    }

    @Override
    public void startSimulation() {
        logger.info("Network Manager đã sẵn sàng.");
    }
}
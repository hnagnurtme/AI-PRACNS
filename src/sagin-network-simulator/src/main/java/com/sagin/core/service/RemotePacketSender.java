package com.sagin.core.service;

import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.PacketTransferWrapper;
import com.sagin.util.NetworkUtils;
import com.sagin.util.PacketSerializerHelper;

import java.io.PrintWriter;
import java.net.Socket;

/**
 * Xử lý việc gửi gói tin serialized (JSON) tới một Node từ xa qua TCP/Socket.
 * Đây là cơ chế RPC giữa các JVM.
 */
public class RemotePacketSender {
    
    private static final int HEALTH_CHECK_TIMEOUT_MS = 500; 
    /**
     * Phương thức tĩnh để gửi gói tin JSON qua Socket.
     * Nó đóng gói Packet và LinkMetric vào một Wrapper trước khi serialize.
     */
    public static boolean sendPacketViaSocket(Packet packet, NodeInfo destInfo, LinkMetric linkMetric) {
        System.out.println("Preparing to send packet " + destInfo.toString());
        String hostName = destInfo.getHost();
        int port = destInfo.getPort();
        
        // 1. ĐÓNG GÓI PACKET VÀ LINK METRIC VÀO WRAPPER
        PacketTransferWrapper wrapper = new PacketTransferWrapper(packet, linkMetric);
    
        // 2. TUẦN TỰ HÓA ĐỐI TƯỢNG WRAPPER
        // Sử dụng phương thức serialize chung để chuyển đổi Wrapper thành JSON String
        String jsonString = PacketSerializerHelper.serialize(wrapper);
        
        if (jsonString == null) {
            // Lỗi đã được log chi tiết trong PacketSerializerHelper
            System.err.printf("[RPC] LỖI: Không thể serialize gói %s (Lỗi JSON).%n", packet.getPacketId());
            return false;
        }

        // --- (Tích hợp Health Check nếu cần thiết, như đã thống nhất) ---
        if (!NetworkUtils.isServiceAvailable(hostName, port, HEALTH_CHECK_TIMEOUT_MS)) {
            System.err.printf("[RPC] LỖI GIAO TIẾP: Node %s:%d KHÔNG PHẢN HỒI (OFFLINE).%n", hostName, port);
            return false;
        }
        
        // 3. GỬI JSON QUA SOCKET
        try (
            // Kết nối đến Hostname (tên Service Docker) và Cổng (Port)
            Socket socket = new Socket(hostName, port);
            PrintWriter writer = new PrintWriter(socket.getOutputStream(), true) 
        ) {
            // Gửi dữ liệu JSON (được đọc bởi TcpGatewayService của Node đích)
            writer.println(jsonString);
            
            // Log thông báo DEBUG cho việc gửi RPC
            // System.out.printf("[RPC] Gửi gói %s thành công tới %s:%d.%n", packet.getPacketId(), hostName, port);
            return true;
            
        } catch (Exception e) {
            // Lỗi kết nối (Connection refused) thường là do Node đích chưa khởi động
            System.err.printf("[RPC] LỖI GIAO TIẾP: Gói %s thất bại khi gửi tới %s:%d. Nguyên nhân: %s%n", 
                                packet.getPacketId(), hostName, port, e.getMessage());
            return false; 
        }
    }
}
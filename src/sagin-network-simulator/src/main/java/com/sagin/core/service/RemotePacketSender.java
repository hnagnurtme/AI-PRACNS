package com.sagin.core.service;

import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.util.PacketSerializerHelper;

import java.io.PrintWriter;
import java.net.Socket;

/**
 * Xử lý việc gửi gói tin serialized (JSON) tới một Node từ xa qua TCP/Socket.
 * Đây là cơ chế RPC giữa các JVM.
 */
public class RemotePacketSender {
    
    // Phương thức tĩnh để gửi gói tin JSON qua Socket
    public static boolean sendPacketViaSocket(Packet packet, NodeInfo destInfo, LinkMetric linkMetric) {
        
        String hostName = destInfo.getHost();
        int port = destInfo.getPort();
        
        // Cần đảm bảo rằng gói tin có ID node hiện tại để nó được xử lý đúng ở node đích
        // Vì đây là cuộc gọi cuối cùng trước khi gói tin đến node đích, 
        // ta giả định packet.setCurrentHoldingNodeId() đã được gọi trước đó.
        
        String jsonString = PacketSerializerHelper.serialize(packet);
        if (jsonString == null) {
            System.err.printf("[RPC] LỖI: Không thể serialize gói %s.%n", packet.getPacketId());
            return false;
        }

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
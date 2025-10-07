package com.sagin.core;

import com.sagin.model.NodeInfo;

/**
 * Interface định nghĩa khả năng lắng nghe các yêu cầu từ bên ngoài 
 * Đưa gói tin vào NodeService.
 */
public interface INodeGatewayService {
    void setNodeServiceReference(INodeService service); 
    /**
     * Khởi động máy chủ lắng nghe tại một cổng nhất định.
     * @param info NodeInfo của node hiện tại (để biết NodeId của mình).
     * @param port Cổng lắng nghe (ví dụ: 8080).
     */
    void startListening(NodeInfo info, int port);

    /**
     * Dừng máy chủ lắng nghe.
     */
    void stopListening();
}
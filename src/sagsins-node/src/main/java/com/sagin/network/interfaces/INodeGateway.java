package com.sagin.network.interfaces;

import java.io.IOException;

import com.sagin.model.NodeInfo;

public interface INodeGateway  {
    /**
     * Khởi động máy chủ lắng nghe tại một cổng nhất định.
     * @param info NodeInfo của node hiện tại (để biết NodeId của mình).
     * @param port Cổng lắng nghe (ví dụ: 8080).
     */
    void startListening(NodeInfo info, int port) throws IOException;

    /**
     * Dừng máy chủ lắng nghe.
     */
    void stopListening();
}

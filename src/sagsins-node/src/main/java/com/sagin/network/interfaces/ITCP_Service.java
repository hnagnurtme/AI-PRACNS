package com.sagin.network.interfaces;

import com.sagin.model.Packet;

public interface ITCP_Service {
    /**
     * Xử lý một packet vừa nhận từ mạng.
     * @param packet Packet đã được deserialize
     */
    void receivePacket(Packet packet);

    /**
     * Gửi một packet (node-to-node) vào hàng đợi.
     * @param packet Packet để gửi (đã chứa nextHopNodeId)
     * @param senderNodeId Node HIỆN TẠI đang gửi packet này (để hạch toán TX)
     */
    void sendPacket(Packet packet, String senderNodeId);
}
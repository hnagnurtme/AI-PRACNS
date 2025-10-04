package com.sagin.core;

import com.sagin.model.Packet;

public interface INodeService {

    /** Khởi tạo và bắt đầu luồng mô phỏng chính cho Node. */
    void startSimulationLoop();

    /** Nhận gói tin đến từ một node láng giềng. */
    void receivePacket(Packet packet);

    /** Quyết định node kế tiếp (Next Hop). */
    String decideNextHop(Packet packet);

    /** Gửi gói tin đã được định tuyến đến node láng giềng. */
    void sendPacket(Packet packet, String nextHopId);

    /** Cập nhật định kỳ trạng thái của node (vị trí, tài nguyên, link láng giềng). */
    void updateNodeState();
}
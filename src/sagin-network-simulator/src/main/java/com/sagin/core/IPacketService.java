package com.sagin.core;

import com.sagin.model.Packet;

/**
 * Interface cho dịch vụ tạo các đối tượng Packet mới theo yêu cầu.
 */
public interface IPacketService {

    /**
     * Tạo một Packet mới dựa trên các tham số dịch vụ TCP/IP giả định.
     * @param sourceId ID Node/User khởi tạo (Node đang chạy).
     * @param destId ID Node/User đích cuối cùng.
     * @param serviceType Loại dịch vụ (ví dụ: VIDEO, VOICE, DATA_BULK).
     * @return Đối tượng Packet đã được khởi tạo.
     */
    Packet generatePacket(String sourceId, String destId, String serviceType);
}
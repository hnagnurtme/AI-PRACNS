package com.sagin.satellite.service;

import com.sagin.satellite.model.Packet;
import com.sagin.satellite.common.SatelliteException;

/**
 * IPacketForwardingService xử lý việc chuyển tiếp gói tin giữa các node
 */
public interface IPacketForwardingService {

    /**
     * Chuyển tiếp packet tới node tiếp theo
     *
     * @param packet Packet cần chuyển tiếp
     * @throws SatelliteException.SendException Nếu không thể gửi packet
     * @throws SatelliteException.InvalidPacketException Nếu packet không hợp lệ
     */
    void forwardPacket(Packet packet) throws SatelliteException.SendException, SatelliteException.InvalidPacketException;

    /**
     * Xử lý packet khi đến node đích cuối cùng
     *
     * @param packet Packet đã đến đích
     */
    void deliverPacket(Packet packet);

    /**
     * Kiểm tra xem packet có đến đích cuối cùng chưa
     *
     * @param packet Packet cần kiểm tra
     * @param currentNodeId ID của node hiện tại
     * @return true nếu đã đến đích, false nếu cần chuyển tiếp
     */
    boolean isDestinationReached(Packet packet, String currentNodeId);

    /**
     * Cập nhật thông tin routing cho packet
     *
     * @param packet Packet cần cập nhật
     * @param currentNodeId Node hiện tại
     * @return true nếu tìm được next hop, false nếu không
     */
    boolean updatePacketRouting(Packet packet, String currentNodeId);

    /**
     * Xử lý packet bị drop (TTL hết hoặc không tìm được đường)
     *
     * @param packet Packet bị drop
     * @param reason Lý do drop
     */
    void handleDroppedPacket(Packet packet, String reason);
}
package com.sagin.core;

import com.sagin.model.ServiceQoS;
import com.sagin.model.Packet;

/**
 * Quản lý thông tin người dùng/ứng dụng và ánh xạ yêu cầu gói tin sang ServiceQoS.
 */
public interface IUserService {

    /**
     * Lấy yêu cầu QoS (ServiceQoS) cho một gói tin cụ thể.
     * @param packet Gói tin.
     * @return ServiceQoS yêu cầu bởi loại dịch vụ của gói tin.
     */
    ServiceQoS getQoSForPacket(Packet packet);
    
    /**
     * Kiểm tra xem người dùng/ứng dụng đích có đang hoạt động không.
     * @param destinationUserId ID người dùng cuối cùng.
     * @return true nếu đích đến khả dụng.
     */
    boolean isUserDestinationAvailable(String destinationUserId);
}
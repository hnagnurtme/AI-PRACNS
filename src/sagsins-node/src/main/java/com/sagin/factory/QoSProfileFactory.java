package com.sagin.factory;

import com.sagin.model.ServiceQoS;
import com.sagin.model.ServiceType;

import java.util.Map;

public final class QoSProfileFactory {

    private static final Map<ServiceType, ServiceQoS> PROFILES = Map.of(
        ServiceType.VIDEO_STREAM, new ServiceQoS(ServiceType.VIDEO_STREAM, 1, 150.0, 30.0, 5.0, 0.01),
        ServiceType.AUDIO_CALL, new ServiceQoS(ServiceType.AUDIO_CALL, 2, 80.0, 10.0, 0.5, 0.005),
        ServiceType.IMAGE_TRANSFER, new ServiceQoS(ServiceType.IMAGE_TRANSFER, 3, 500.0, 100.0, 1.0, 0.02),
        ServiceType.FILE_TRANSFER, new ServiceQoS(ServiceType.FILE_TRANSFER, 4, 2000.0, 500.0, 2.0, 0.05),
        ServiceType.TEXT_MESSAGE, new ServiceQoS(ServiceType.TEXT_MESSAGE, 5, 1000.0, 200.0, 0.1, 0.01)
    );

    // Private constructor để ngăn việc tạo instance
    private QoSProfileFactory() {}

    /**
     * Trả về một cấu hình ServiceQoS dựa trên loại dịch vụ.
     * @param serviceType Loại dịch vụ cần lấy cấu hình.
     * @return Đối tượng ServiceQoS tương ứng.
     * @throws IllegalArgumentException nếu không tìm thấy cấu hình.
     */
    public static ServiceQoS getQosProfile(ServiceType serviceType) {
        ServiceQoS profile = PROFILES.get(serviceType);
        if (profile == null) {
            throw new IllegalArgumentException("No QoS profile found for service type: " + serviceType);
        }
        return profile;
    }
}
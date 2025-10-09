package com.sagin.core.service;
import com.sagin.core.IUserService;
import com.sagin.model.ServiceQoS;
import com.sagin.model.Packet;
import java.util.HashMap;
import java.util.Map;

public class UserService implements IUserService {

    // Map chứa cấu hình ServiceQoS cho từng loại dịch vụ
    private final Map<String, ServiceQoS> qosConfig = new HashMap<>();

    // Map mô phỏng trạng thái người dùng cuối cùng (User Registry)
    private final Map<String, Boolean> userAvailability = new HashMap<>();

    public UserService() {
        // Khởi tạo cấu hình QoS mặc định (Dữ liệu mẫu)
        initializeQoSConfig();
        // Khởi tạo trạng thái người dùng mẫu
        initializeUserRegistry();
    }
    
    // --- Khởi tạo Cấu hình Dữ liệu (Mock Data) ---
    private void initializeQoSConfig() {
        // VOIP/Real-time: Ưu tiên cao, độ trễ và Jitter thấp, BW thấp.
        qosConfig.put("VOICE", 
            new ServiceQoS("VOICE", 1, 150.0, 30.0, 0.128, 0.01)); 

        // Video Streaming: Ưu tiên trung bình, BW cao, chấp nhận độ trễ trung bình.
        qosConfig.put("VIDEO", 
            new ServiceQoS("VIDEO", 2, 500.0, 80.0, 5.0, 0.02));
            
        // Background Data: Ưu tiên thấp, chấp nhận độ trễ và mất gói cao.
        qosConfig.put("DATA", 
            new ServiceQoS("DATA", 5, 2000.0, 500.0, 0.05, 0.05));
    }

    private void initializeUserRegistry() {
        // Đích đến mẫu có sẵn
        userAvailability.put("USER_A", true);
        userAvailability.put("USER_B", true);
        userAvailability.put("USER_C", false); 
    }


    // --- Triển khai Interface ---

    /**
     * @inheritdoc
     */
    @Override
    public ServiceQoS getQoSForPacket(Packet packet) {
        String serviceType = packet.getServiceType();
        ServiceQoS qos = qosConfig.get(serviceType);

        // Trả về QoS mặc định (ví dụ: DATA) nếu loại dịch vụ không được xác định
        if (qos == null) {
            return qosConfig.get("DATA"); 
        }
        return qos;
    }

    /**
     * @inheritdoc
     */
    @Override
    public boolean isUserDestinationAvailable(String destinationUserId) {
        // Trong môi trường thực tế, đây sẽ là một cuộc gọi API tới một dịch vụ hiện diện (Presence Service)
        return userAvailability.getOrDefault(destinationUserId, false);
    }
}
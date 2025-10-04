package com.sagin.core.service;

import com.sagin.core.IPacketService;
import com.sagin.model.Packet;
import com.sagin.model.ServiceQoS;
import com.sagin.util.ProjectConstant;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Triển khai dịch vụ tạo gói tin, gán các thuộc tính cần thiết như QoS và ID.
 */
public class PacketService implements IPacketService {

    // Dùng AtomicInteger để đảm bảo bộ đếm gói tin an toàn đa luồng
    private final AtomicInteger packetCounter = new AtomicInteger(0);

    @Override
    public Packet generatePacket(String sourceId, String destId, String serviceType) {
        
        // Lấy cấu hình QoS yêu cầu
        ServiceQoS qosConfig = getQoSConfig(serviceType);

        Packet packet = new Packet();
        packet.setPacketId(sourceId + "-" + packetCounter.incrementAndGet()); // ID duy nhất
        packet.setSourceUserId(sourceId); 
        packet.setDestinationUserId(destId);
        packet.setServiceType(serviceType);
        
        // Thiết lập các trường routing và QoS
        packet.setPayloadSizeByte(getPayloadSize(serviceType));
        packet.setPriorityLevel(qosConfig.getDefaultPriority());
        packet.setTTL(ProjectConstant.DEFAULT_TTL);
        packet.setTimestamp(System.currentTimeMillis());
        packet.setMaxAcceptableLatencyMs(qosConfig.getMaxLatencyMs());
        packet.setMaxAcceptableLossRate(qosConfig.getMaxLossRate());
        
        return packet;
    }
    
    /**
     * Phương thức giả lập lấy cấu hình QoS (thường là một Service/Factory riêng).
     */
    private ServiceQoS getQoSConfig(String serviceType) {
        switch (serviceType) {
            case ProjectConstant.SERVICE_TYPE_VOICE:
                // Voice: Độ trễ thấp, ưu tiên cao
                return new ServiceQoS(serviceType, 1, 150.0, 0.03); 
            case ProjectConstant.SERVICE_TYPE_VIDEO:
                // Video: Băng thông cao, độ trễ chấp nhận được
                return new ServiceQoS(serviceType, 2, 400.0, 0.05); 
            case ProjectConstant.SERVICE_TYPE_CONTROL:
                // Control: Ưu tiên cao nhất
                return new ServiceQoS(serviceType, 0, 50.0, 0.001); 
            case ProjectConstant.SERVICE_TYPE_DATA_BULK:
            default:
                // Data Bulk: Ưu tiên thấp
                return new ServiceQoS(serviceType, 3, 1500.0, 0.1); 
        }
    }

    /** Gán kích thước Payload dựa trên loại dịch vụ. */
    private int getPayloadSize(String serviceType) {
        switch (serviceType) {
            case ProjectConstant.SERVICE_TYPE_VOICE:
                return 1500; 
            case ProjectConstant.SERVICE_TYPE_VIDEO:
                return 60000; 
            default:
                return 8000;
        }
    }
}
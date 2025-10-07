package com.sagin.core.service;

import com.sagin.core.ILinkManagerService;
import com.sagin.model.Geo3D;
import com.sagin.model.LinkAction;
import com.sagin.model.LinkMetric;
import com.sagin.util.ProjectConstant;

// Đổi tên class thành LinkManagerService và implement ILinkManagerService
public class LinkManagerService implements ILinkManagerService { 

    @Override
    public LinkMetric calculateInitialMetric(Geo3D sourcePos, Geo3D destPos) {
        double distance = sourcePos.distanceTo(destPos);
        
        LinkMetric metric = new LinkMetric();
        metric.setSourceNodeId("N/A");
        metric.setDestinationNodeId("N/A");
        metric.setDistanceKm(distance);
        metric.setLinkActive(true);
        
        // Tính độ trễ truyền dẫn
        double propagationLatencyMs = distance / ProjectConstant.SPEED_OF_LIGHT_KM_PER_MS;
        metric.setLatencyMs(propagationLatencyMs);

        // Đặt các giá trị mặc định/max
        metric.setMaxBandwidthMbps(1000.0);
        metric.setCurrentAvailableBandwidthMbps(1000.0);
        metric.setPacketLossRate(0.01);
        
        // Cần tính toán linkScore ban đầu
        // metric.calculateLinkScore(); // (Nếu bạn thêm method này vào LinkMetric)
        
        return metric;
    }

    @Override
    public LinkMetric applyLinkAction(LinkMetric currentMetric, LinkAction action) {
        // --- Logic mô phỏng RL/Vật lý ---
        
        // 1. Công suất ảnh hưởng đến Loss Rate
        double powerEffect = (action.getTransmitPowerDbm() - 30.0) / 10.0;
        currentMetric.setPacketLossRate(Math.max(0.001, currentMetric.getPacketLossRate() - powerEffect * 0.005));
        
        // 2. Điều chế ảnh hưởng đến Băng thông khả dụng
        double modulationFactor = action.getModulationScheme() * 0.25; 
        currentMetric.setCurrentAvailableBandwidthMbps(currentMetric.getMaxBandwidthMbps() * modulationFactor);

        currentMetric.setLastUpdated(System.currentTimeMillis());
        return currentMetric;
    }

    @Override
    public LinkMetric updateDynamicMetrics(LinkMetric currentMetric) {
        // Ví dụ: Mô phỏng nhiễu môi trường
        double randomNoise = (Math.random() * 5.0); 
        currentMetric.setLatencyMs(currentMetric.getLatencyMs() + randomNoise);
        currentMetric.setLastUpdated(System.currentTimeMillis());
        return currentMetric;
    }
}
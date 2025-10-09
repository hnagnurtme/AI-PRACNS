package com.sagin.core.service;

import com.sagin.core.ILinkManagerService;
import com.sagin.model.*;
import com.sagin.util.GeoUtils;

import java.util.Random;

public class LinkManagerService implements ILinkManagerService {

    private final Random random = new Random();
    private static final double BASE_MAX_BANDWIDTH_MBPS = 1000.0; // Mbps cơ bản cho liên kết vệ tinh
    private static final double WEATHER_IMPACT_FACTOR = 0.1; // Hệ số ảnh hưởng của thời tiết

    /**
     * @inheritdoc
     */
    @Override
    public boolean checkVisibility(NodeInfo sourceNode, NodeInfo destNode) {
        // Chỉ cần gọi hàm tiện ích đã định nghĩa
        if (sourceNode == null || destNode == null)
            return false;

        // Kiểm tra tầm nhìn thực tế (phải là Line of Sight)
        boolean isVisible = GeoUtils.checkVisibility(sourceNode.getPosition(), destNode.getPosition());

        // Kiểm tra logic bổ sung: Trạm mặt đất có đang bị bão quá lớn không?
        if (destNode.getNodeType() == NodeType.GROUND_STATION &&
                destNode.getWeather() == WeatherCondition.SEVERE_STORM) {
            return false;
        }

        return isVisible;
    }

    /**
     * @inheritdoc
     */
    @Override
    public LinkMetric calculateLinkMetric(NodeInfo sourceNode, NodeInfo destNode) {
        LinkMetric metric = new LinkMetric();
        metric.setSourceNodeId(sourceNode.getNodeId());
        metric.setDestinationNodeId(destNode.getNodeId());

        // 1. Tính toán Vật lý
        double distanceKm = GeoUtils.calculateDistance3D(sourceNode.getPosition(), destNode.getPosition());
        metric.setDistanceKm(distanceKm);

        // Độ trễ truyền dẫn (Propagation Delay)
        double propagationDelayMs = GeoUtils.calculatePropagationDelayMs(distanceKm);
        metric.setLatencyMs(propagationDelayMs);

        // 2. Tính toán Suy hao (Attenuation)
        double totalAttenuationDb = 0.0;

        // Suy hao khí quyển từ Ground Station
        if (sourceNode.getNodeType() == NodeType.GROUND_STATION) {
            totalAttenuationDb += sourceNode.getWeather().getTypicalAttenuationDb();
        }
        if (destNode.getNodeType() == NodeType.GROUND_STATION) {
            totalAttenuationDb += destNode.getWeather().getTypicalAttenuationDb();
        }
        metric.setLinkAttenuationDb(totalAttenuationDb);

        // 3. Tính toán Băng thông (Base Max BW)
        // Giả định: Liên kết LEO-GS có BW cao hơn LEO-LEO
        if (sourceNode.getNodeType() == NodeType.GROUND_STATION || destNode.getNodeType() == NodeType.GROUND_STATION) {
            metric.setMaxBandwidthMbps(BASE_MAX_BANDWIDTH_MBPS * 1.5); // Liên kết GS thường mạnh hơn
        } else {
            metric.setMaxBandwidthMbps(BASE_MAX_BANDWIDTH_MBPS); // Liên kết ISL
        }

        // Giả định: Hiện tại 80% BW khả dụng (sẽ được cập nhật động sau)
        metric.setCurrentAvailableBandwidthMbps(metric.getMaxBandwidthMbps() * 0.8);

        // 4. Tính toán Mất gói (Packet Loss Rate)
        double baseLoss = 0.001;
        // Suy hao khí quyển làm tăng mất gói
        metric.setPacketLossRate(baseLoss + totalAttenuationDb * WEATHER_IMPACT_FACTOR);

        // 5. Kiểm tra trạng thái Link và Tính Link Score
        metric.setLinkActive(checkVisibility(sourceNode, destNode));
        metric.calculateLinkScore(); // Tính toán Link Score (đã có logic tối ưu trong LinkMetric)

        metric.setLastUpdated(System.currentTimeMillis());
        return metric;
    }

    /**
     * @inheritdoc
     *             Mô phỏng tác động của hành động điều khiển (ví dụ: Beam Steering,
     *             Power Boost)
     */
    @Override
    public LinkMetric applyLinkAction(LinkMetric currentMetric, LinkAction action) {
        // Trong môi phỏng thực tế, logic phức tạp sẽ được đặt ở đây.
        switch (action) {
            case POWER_BOOST:
                // Tăng cường độ truyền: giảm Packet Loss và tăng BW khả dụng tạm thời
                currentMetric.setPacketLossRate(currentMetric.getPacketLossRate() * 0.5);
                currentMetric.setCurrentAvailableBandwidthMbps(currentMetric.getCurrentAvailableBandwidthMbps() * 1.2);
                break;
            case BEAM_STEERING:
                // Tăng cường BW khả dụng bằng cách tập trung chùm tia
                currentMetric.setCurrentAvailableBandwidthMbps(currentMetric.getCurrentAvailableBandwidthMbps() * 1.1);
                break;
            case ADJUST_FREQUENCY:
                // Giảm nhiễu/giao thoa: Giảm mất gói
                currentMetric.setPacketLossRate(Math.max(0.0001, currentMetric.getPacketLossRate() * 0.8));
                break;
        }
        currentMetric.calculateLinkScore();
        return currentMetric;
    }

    /**
     * @inheritdoc
     *             Cập nhật các yếu tố ngẫu nhiên (chẳng hạn như nhiễu hoặc tắc
     *             nghẽn ngẫu nhiên).
     */
    @Override
    public LinkMetric updateDynamicMetrics(LinkMetric currentMetric) {
        // 1. Mô phỏng nhiễu/biến thiên ngẫu nhiên (Random Fluctuation)
        double noiseFactor = 1.0 + (random.nextDouble() - 0.5) * 0.05; // Biến thiên 5%

        // Tỷ lệ mất gói ngẫu nhiên
        currentMetric.setPacketLossRate(
                Math.min(0.1, currentMetric.getPacketLossRate() * noiseFactor));

        // Độ trễ ngẫu nhiên (Micro-delay do giao thoa)
        currentMetric.setLatencyMs(
                currentMetric.getLatencyMs() + (random.nextDouble() * 0.1)); // Thêm độ trễ ngẫu nhiên < 0.1 ms

        // 2. Cập nhật Link Score và Last Updated
        currentMetric.calculateLinkScore();
        currentMetric.setLastUpdated(System.currentTimeMillis());

        return currentMetric;
    }
}
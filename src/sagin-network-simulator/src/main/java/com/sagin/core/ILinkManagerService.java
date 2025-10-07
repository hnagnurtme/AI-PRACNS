package com.sagin.core;

import com.sagin.model.Geo3D;
import com.sagin.model.LinkAction;
import com.sagin.model.LinkMetric;

/**
 * Interface cho dịch vụ quản lý logic vật lý và hiệu suất của các đường truyền (Link).
 */
public interface ILinkManagerService { // Đổi tên class thành ILinkManagerService

    /**
     * Tính toán LinkMetric ban đầu dựa trên vị trí của hai node.
     */
    LinkMetric calculateInitialMetric(Geo3D sourcePos, Geo3D destPos);

    /**
     * Mô phỏng tác động của hành động điều khiển (LinkAction) lên chất lượng link.
     */
    LinkMetric applyLinkAction(LinkMetric currentMetric, LinkAction action);

    /**
     * Cập nhật định kỳ các yếu tố động như mất gói và độ trễ dựa trên điều kiện môi trường.
     */
    LinkMetric updateDynamicMetrics(LinkMetric currentMetric);
}
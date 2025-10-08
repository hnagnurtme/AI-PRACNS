package com.sagin.core;

import com.sagin.model.NodeInfo;
import com.sagin.model.LinkMetric;
import com.sagin.model.LinkAction;

/**
 * Quản lý logic vật lý, tầm nhìn, và hiệu suất của các đường truyền (Link Metric).
 */
public interface ILinkManagerService { 

    /**
     * Kiểm tra xem có liên kết vật lý (Line of Sight) giữa hai node hay không.
     */
    boolean checkVisibility(NodeInfo sourceNode, NodeInfo destNode);
    
    /**
     * Tính toán LinkMetric giữa hai node, bao gồm suy hao thời tiết.
     * @param sourceNode NodeInfo của node nguồn (để lấy thông tin vệ tinh/mặt đất).
     * @param destNode NodeInfo của node đích.
     * @return LinkMetric được tính toán.
     */
    LinkMetric calculateLinkMetric(NodeInfo sourceNode, NodeInfo destNode);

    /**
     * Mô phỏng tác động của hành động điều khiển (LinkAction) lên chất lượng link.
     */
    LinkMetric applyLinkAction(LinkMetric currentMetric, LinkAction action);

    /**
     * Cập nhật định kỳ các yếu tố động (chẳng hạn như di chuyển vệ tinh, nhiễu).
     */
    LinkMetric updateDynamicMetrics(LinkMetric currentMetric);
}
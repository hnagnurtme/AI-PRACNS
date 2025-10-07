package com.sagin.repository; 

import com.sagin.model.NodeInfo;
import java.util.Map;

/**
 * Interface Repository để truy cập thông tin Node từ nguồn dữ liệu (Database).
 */
public interface INodeRepository {

    /**
     * Tải tất cả NodeInfo từ nguồn dữ liệu (ví dụ: Firebase) để khởi tạo mạng.
     * @return Map<NodeId, NodeInfo> của toàn bộ mạng lưới.
     */
    Map<String, NodeInfo> loadAllNodeConfigs();

    /**
     * Cập nhật thông tin của một Node (ví dụ: vị trí, trạng thái) lên Database.
     * @param nodeId ID của Node cần cập nhật.
     * @param info Dữ liệu NodeInfo mới.
     */
    void updateNodeInfo(String nodeId, NodeInfo info);
}
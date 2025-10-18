package com.sagin.repository; 

import com.sagin.model.NodeInfo;

import java.util.Collection;
import java.util.Map;
import java.util.Optional;

/**
 * Interface Repository để truy cập thông tin Node từ nguồn dữ liệu (Database).
 */
public interface INodeRepository {

    /**
     * Tải tất cả NodeInfo từ nguồn dữ liệu để khởi tạo mạng.
     * @return Map<NodeId, NodeInfo> của toàn bộ mạng lưới.
     */
    Map<String, NodeInfo> loadAllNodeConfigs();

    /**
     * Cập nhật thông tin của một Node (ví dụ: vị trí, trạng thái) lên Database.
     * @param nodeId ID của Node cần cập nhật.
     * @param info Dữ liệu NodeInfo mới.
     */
    void updateNodeInfo(String nodeId, NodeInfo info);


    /**
     * Tải thông tin NodeInfo của một Node cụ thể từ Database.
     * @param nodeId ID cua Node cần tra cứu.
     * @return NodeInfo của Node đó, hoặc null nếu không tìm thấy.
     */
    Optional<NodeInfo> getNodeInfo(String nodeId);


    void bulkUpdateNodes(Collection<NodeInfo> nodes);

}
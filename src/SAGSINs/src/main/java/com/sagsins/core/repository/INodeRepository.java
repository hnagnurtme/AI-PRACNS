package com.sagsins.core.repository;

import com.sagsins.core.model.NodeInfo;
import java.util.List;
import java.util.Optional;

/**
 * Interface Repository định nghĩa các thao tác CRUD cơ bản cho NodeInfo.
 * Đây là giao diện chuẩn hóa tầng truy cập dữ liệu (Data Access Layer).
 */
public interface INodeRepository {

    // --- CREATE / UPDATE ---
    /**
     * Lưu trữ một NodeInfo mới hoặc cập nhật một NodeInfo đã tồn tại.
     * @param node Đối tượng NodeInfo cần lưu.
     * @return NodeInfo đã được lưu/cập nhật (thường kèm theo ID được gán).
     */
    NodeInfo save(NodeInfo node);

    // --- READ ---
    /**
     * Tìm kiếm một NodeInfo theo ID.
     * @param nodeId ID định danh của Node.
     * @return Optional chứa NodeInfo nếu tìm thấy, Optional rỗng nếu không.
     */
    Optional<NodeInfo> findById(String nodeId);

    /**
     * Lấy tất cả các NodeInfo hiện có.
     * @return Danh sách tất cả NodeInfo.
     */
    List<NodeInfo> findAll();

    // --- DELETE ---
    /**
     * Xóa một NodeInfo dựa trên ID.
     * @param nodeId ID của NodeInfo cần xóa.
     */
    void deleteById(String nodeId);

    /**
     * Kiểm tra sự tồn tại của một NodeInfo bằng ID.
     * @param nodeId ID của NodeInfo.
     * @return true nếu Node tồn tại, false nếu ngược lại.
     */
    boolean existsById(String nodeId);
}
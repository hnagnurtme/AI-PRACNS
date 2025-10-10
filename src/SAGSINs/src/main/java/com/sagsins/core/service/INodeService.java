// src/main/java/com/sagsins/core/service/INodeService.java
package com.sagsins.core.service;

import java.util.List;
import java.util.Optional;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.DTOs.response.DockerResposne;

/**
 * Interface định nghĩa các nghiệp vụ (Service Operations) cho việc quản lý Node (CRUD).
 */
public interface INodeService {
    
    // --- CREATE ---
    /**
     * Tạo một Node mới dựa trên yêu cầu DTO.
     * @param request DTO chứa dữ liệu cần thiết để tạo Node.
     * @return NodeInfo đã được tạo và lưu trữ.
     */
    NodeDTO createNode(CreateNodeRequest request);

    // --- READ ---
    /**
     * Lấy danh sách tất cả các Node đang tồn tại.
     * @return Danh sách các NodeInfo.
     */
    List<NodeDTO> getAllNodes();

    /**
     * Lấy chi tiết một Node dựa trên ID.
     * @param nodeId ID của Node cần tìm.
     * @return Optional chứa NodeInfo nếu tìm thấy, hoặc Optional rỗng.
     */
    Optional<NodeDTO> getNodeById(String nodeId);

    // --- UPDATE ---
    /**
     * Cập nhật thông tin của một Node đã tồn tại.
     * @param nodeId ID của Node cần cập nhật.
     * @param request DTO chứa các trường dữ liệu mới (có thể là một phần).
     * @return Optional chứa NodeInfo đã được cập nhật nếu thành công, hoặc Optional rỗng nếu không tìm thấy ID.
     */
    Optional<NodeDTO> updateNode(String nodeId, UpdateNodeRequest request);

    // --- DELETE ---
    /**
     * Xóa một Node khỏi hệ thống.
     * @param nodeId ID của Node cần xóa.
     * @return true nếu xóa thành công, false nếu Node không tồn tại.
     */
    boolean deleteNode(String nodeId);


    boolean runNodeProcess(String nodeId);
}
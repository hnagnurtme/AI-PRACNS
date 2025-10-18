package com.sagsins.core.service;

import java.util.List;
import java.util.Optional;

import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.DTOs.request.UpdateStatusRequest;


/**
 * Interface định nghĩa các nghiệp vụ (Service Operations) cho việc quản lý Node (CRUD).
 */
public interface INodeService {
    
    // --- READ ---
    /**
     * Lấy danh sách tất cả các Node đang tồn tại.
     * @return Danh sách các NodeDTO.
     */
    List<NodeDTO> getAllNodes();

    /**
     * Lấy chi tiết một Node dựa trên ID.
     * @param nodeId ID của Node cần tìm.
     * @return Optional chứa NodeDTO nếu tìm thấy, hoặc Optional rỗng.
     */
    Optional<NodeDTO> getNodeById(String nodeId);

    // --- UPDATE ---
    /**
     * Cập nhật thông tin một Node (partial update).
     * @param nodeId ID của Node cần cập nhật.
     * @param request Request chứa các thông tin cần cập nhật.
     * @return NodeDTO đã được cập nhật.
     * @throws com.sagsins.core.exception.NotFoundException nếu không tìm thấy node.
     */
    NodeDTO updateNodeStatus(String nodeId, UpdateStatusRequest request);

}
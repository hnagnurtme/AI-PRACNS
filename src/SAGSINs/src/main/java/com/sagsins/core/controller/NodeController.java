package com.sagsins.core.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.DTOs.request.UpdateStatusRequest;
import com.sagsins.core.exception.NotFoundException;
import com.sagsins.core.service.INodeService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/v1")
@Tag(
    name = "Node Controller",
    description = "Quản lý các node trong hệ thống SAGSINS — tạo, cập nhật, xóa, và khởi chạy node mô phỏng."
)
public class NodeController {

    private final INodeService nodeService;

    public NodeController(INodeService nodeService) {
        this.nodeService = nodeService;
    }

    // ---------------- GET ALL NODES ----------------
    @Operation(
        summary = "Lấy danh sách tất cả node",
        description = "Trả về danh sách tất cả các node đang được quản lý trong hệ thống SAGSINS.",
        responses = {
            @ApiResponse(
                responseCode = "200",
                description = "Lấy danh sách thành công",
                content = @Content(
                    mediaType = "application/json",
                    array = @ArraySchema(schema = @Schema(implementation = NodeDTO.class))
                )
            )
        }
    )
    @GetMapping("/nodes")
    public ResponseEntity<List<NodeDTO>> getAllNodes() {
        List<NodeDTO> nodes = nodeService.getAllNodes();
        return ResponseEntity.ok(nodes);
    }

    // ---------------- GET NODE BY ID ----------------
    @Operation(
        summary = "Lấy thông tin chi tiết một node",
        description = "Trả về thông tin chi tiết của một node dựa trên ID.",
        responses = {
            @ApiResponse(
                responseCode = "200",
                description = "Lấy thông tin thành công",
                content = @Content(
                    mediaType = "application/json",
                    schema = @Schema(implementation = NodeDTO.class)
                )
            ),
            @ApiResponse(
                responseCode = "404",
                description = "Không tìm thấy node với ID được cung cấp"
            )
        }
    )
    @GetMapping("/nodes/{id}")
    public ResponseEntity<NodeDTO> getNodeById(
            @Parameter(description = "ID của node cần lấy thông tin", required = true)
            @PathVariable("id") String id) {
        return nodeService.getNodeById(id)
                .map(ResponseEntity::ok)
                .orElseThrow(() -> new NotFoundException("Node not found with ID: " + id));
    }

    // ---------------- UPDATE NODE STATUS (PATCH) ----------------
    @Operation(
        summary = "Cập nhật trạng thái và thông tin node (partial update)",
        description = "Cập nhật một hoặc nhiều trường thông tin của node. Chỉ các trường được cung cấp sẽ được cập nhật.",
        responses = {
            @ApiResponse(
                responseCode = "200",
                description = "Cập nhật thành công",
                content = @Content(
                    mediaType = "application/json",
                    schema = @Schema(implementation = NodeDTO.class)
                )
            ),
            @ApiResponse(
                responseCode = "404",
                description = "Không tìm thấy node với ID được cung cấp"
            ),
            @ApiResponse(
                responseCode = "400",
                description = "Dữ liệu đầu vào không hợp lệ"
            )
        }
    )
    @PatchMapping("/nodes/{id}")
    public ResponseEntity<NodeDTO> updateNodeStatus(
            @Parameter(description = "ID của node cần cập nhật", required = true)
            @PathVariable("id") String id,
            @Parameter(description = "Dữ liệu cập nhật trạng thái (chỉ cần cung cấp các trường cần thay đổi)", required = true)
            @Valid @RequestBody UpdateStatusRequest request) {
        NodeDTO updatedNode = nodeService.updateNodeStatus(id, request);
        return ResponseEntity.ok(updatedNode);
    }
}

package com.sagsins.core.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.service.INodeService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;

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
        return new ResponseEntity<>(nodeService.getAllNodes(), HttpStatus.OK);
    }

    // ---------------- CREATE NODE ----------------
    @Operation(
        summary = "Tạo mới một node",
        description = "Khởi tạo một node mới trong hệ thống với thông tin được truyền qua `CreateNodeRequest`.",
        responses = {
            @ApiResponse(
                responseCode = "201",
                description = "Tạo node thành công",
                content = @Content(schema = @Schema(implementation = NodeDTO.class))
            ),
            @ApiResponse(responseCode = "400", description = "Dữ liệu không hợp lệ", content = @Content)
        }
    )
    @PostMapping("/nodes")
    public ResponseEntity<NodeDTO> createNode(
        @io.swagger.v3.oas.annotations.parameters.RequestBody(
            required = true,
            description = "Thông tin cấu hình node mới cần tạo",
            content = @Content(schema = @Schema(implementation = CreateNodeRequest.class))
        )
        @org.springframework.web.bind.annotation.RequestBody CreateNodeRequest request
    ) {
        return ResponseEntity.status(HttpStatus.CREATED).body(nodeService.createNode(request));
    }

    // ---------------- UPDATE NODE ----------------
    @Operation(
        summary = "Cập nhật thông tin node",
        description = "Cập nhật dữ liệu node dựa theo `nodeId` và request body.",
        responses = {
            @ApiResponse(responseCode = "200", description = "Cập nhật thành công", content = @Content(schema = @Schema(implementation = NodeDTO.class))),
            @ApiResponse(responseCode = "404", description = "Không tìm thấy node", content = @Content)
        }
    )
    @PatchMapping("/nodes/{nodeId}")
    public ResponseEntity<NodeDTO> updateNode(
        @Parameter(description = "ID của node cần cập nhật", example = "NODE_001")
        @PathVariable String nodeId,
        @io.swagger.v3.oas.annotations.parameters.RequestBody(
            required = true,
            description = "Thông tin cập nhật node",
            content = @Content(schema = @Schema(implementation = UpdateNodeRequest.class))
        )
        UpdateNodeRequest request
    ) {
        return nodeService.updateNode(nodeId, request)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    // ---------------- DELETE NODE ----------------
    @Operation(
        summary = "Xóa một node",
        description = "Xóa node khỏi hệ thống dựa theo ID được cung cấp.",
        responses = {
            @ApiResponse(responseCode = "204", description = "Xóa thành công", content = @Content),
            @ApiResponse(responseCode = "404", description = "Không tìm thấy node", content = @Content)
        }
    )
    @DeleteMapping("/nodes/{nodeId}")
    public ResponseEntity<Void> deleteNode(
        @Parameter(description = "ID của node cần xóa", example = "NODE_001")
        @PathVariable String nodeId
    ) {
        boolean deleted = nodeService.deleteNode(nodeId);
        return deleted ? ResponseEntity.noContent().build() : ResponseEntity.notFound().build();
    }

    // ---------------- RUN NODE PROCESS ----------------
    @Operation(
        summary = "Khởi chạy tiến trình của node",
        description = "Kích hoạt node thực thi trong môi trường mô phỏng (ví dụ: chạy Docker container hoặc tiến trình mạng).",
        responses = {
            @ApiResponse(responseCode = "200", description = "Node đã được khởi chạy thành công", content = @Content),
            @ApiResponse(responseCode = "500", description = "Lỗi khi khởi chạy node", content = @Content)
        }
    )
    @PostMapping("/nodes/run/{nodeId}")
    public ResponseEntity<Void> runNodeProcess(
        @Parameter(description = "ID của node cần khởi chạy", example = "NODE_001")
        @PathVariable String nodeId
    ) {
        boolean started = nodeService.runNodeProcess(nodeId);
        return started
            ? ResponseEntity.ok().build()
            : ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
    }
}

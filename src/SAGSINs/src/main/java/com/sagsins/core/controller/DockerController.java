package com.sagsins.core.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.DTOs.response.DockerResposne;
import com.sagsins.core.service.IDockerService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api/v1/docker")
@Tag(
    name = "Docker Controller",
    description = "Quản lý và truy vấn thông tin các container Docker trong hệ thống SAGSINS."
)
public class DockerController {

    private final IDockerService dockerService;

    public DockerController(IDockerService dockerService) {
        this.dockerService = dockerService;
    }

    @Operation(
        summary = "Lấy danh sách container Docker",
        description = "Trả về danh sách tất cả các container Docker đang chạy hoặc đã dừng tùy theo tham số `isRunning`.",
        responses = {
            @ApiResponse(
                responseCode = "200",
                description = "Danh sách container được trả về thành công",
                content = @Content(
                    mediaType = "application/json",
                    array = @ArraySchema(schema = @Schema(implementation = DockerResposne.class))
                )
            ),
            @ApiResponse(responseCode = "500", description = "Lỗi nội bộ hệ thống", content = @Content)
        }
    )
    @GetMapping("/allLinks")
    public ResponseEntity<List<DockerResposne>> getAllEntitys(
        @Parameter(
            description = "Nếu `true` thì chỉ lấy container đang chạy, nếu `false` thì lấy tất cả.",
            required = true,
            example = "true"
        )
        @RequestParam boolean isRunning
    ) {
        List<DockerResposne> containers = dockerService.getAllContainers(isRunning);
        return ResponseEntity.ok(containers);
    }
}

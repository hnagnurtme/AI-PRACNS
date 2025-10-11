package com.sagsins.core.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.DTOs.response.HealthResposne;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@Tag(name = "Core Controller", description = "Kiểm tra trạng thái hoạt động của hệ thống SAGSINS Core")
public class CoreController {

    @Operation(
        summary = "Kiểm tra sức khỏe của server",
        description = "Trả về thông tin xác nhận rằng SAGSINS Core Server đang hoạt động bình thường."
    )
    @ApiResponse(
        responseCode = "200",
        description = "Server đang hoạt động bình thường",
        content = @Content(mediaType = "application/json",
            schema = @Schema(implementation = HealthResposne.class))
    )
    @GetMapping("/health")
    public ResponseEntity<HealthResposne> checkHealth() {
        return new ResponseEntity<>(new HealthResposne("OK", "Server is running"), HttpStatus.OK);
    }
}

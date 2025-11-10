package com.sagsins.core.controller;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.*;

import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.repository.INodeRepository;
import com.sagsins.core.service.SimulationScenarioFactoryService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api/simulation")
@Tag(
    name = "Scenario Controller",
    description = "Quản lý các kịch bản mô phỏng trong hệ thống SAGSINS"
)
public class ScenarioController {

    private static final Logger logger = LoggerFactory.getLogger(ScenarioController.class);
    
    private final SimulationScenarioFactoryService scenarioService;
    private final INodeRepository nodeRepository;
    private final SimpMessagingTemplate messagingTemplate;

    public ScenarioController(
            SimulationScenarioFactoryService scenarioService,
            INodeRepository nodeRepository,
            SimpMessagingTemplate messagingTemplate) {
        this.scenarioService = scenarioService;
        this.nodeRepository = nodeRepository;
        this.messagingTemplate = messagingTemplate;
    }

    @Operation(
        summary = "Lấy danh sách các kịch bản có sẵn",
        description = "Trả về danh sách tất cả các kịch bản mô phỏng có thể áp dụng",
        responses = {
            @ApiResponse(responseCode = "200", description = "Lấy danh sách thành công")
        }
    )
    @GetMapping("/scenarios")
    public ResponseEntity<List<String>> getAvailableScenarios() {
        return ResponseEntity.ok(scenarioService.getAvailableScenarios());
    }

    @Operation(
        summary = "Lấy kịch bản hiện tại",
        description = "Trả về thông tin về kịch bản đang được áp dụng",
        responses = {
            @ApiResponse(responseCode = "200", description = "Lấy thông tin thành công")
        }
    )
    @GetMapping("/scenario/current")
    public ResponseEntity<Map<String, Object>> getCurrentScenario() {
        String currentScenario = scenarioService.getCurrentScenario();
        Map<String, Object> response = new HashMap<>(scenarioService.getScenarioInfo(currentScenario));
        response.put("scenario", currentScenario);
        return ResponseEntity.ok(response);
    }

    @Operation(
        summary = "Áp dụng kịch bản mô phỏng",
        description = "Áp dụng một kịch bản cụ thể cho tất cả các node trong hệ thống",
        responses = {
            @ApiResponse(responseCode = "200", description = "Áp dụng kịch bản thành công"),
            @ApiResponse(responseCode = "400", description = "Tên kịch bản không hợp lệ")
        }
    )
    @PostMapping("/scenario/{scenarioName}")
    public ResponseEntity<Map<String, Object>> applyScenario(
            @Parameter(description = "Tên kịch bản cần áp dụng (NORMAL, WEATHER_EVENT, NODE_OVERLOAD, NODE_OFFLINE, TRAFFIC_SPIKE)", required = true)
            @PathVariable String scenarioName) {
        
        logger.info("Applying scenario: {}", scenarioName);
        
        // Validate scenario name
        if (!scenarioService.getAvailableScenarios().contains(scenarioName)) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("error", "Invalid scenario name: " + scenarioName);
            return ResponseEntity.badRequest().body(errorResponse);
        }
        
        // Set the scenario
        scenarioService.setScenario(scenarioName);
        
        // Apply to all nodes
        List<NodeInfo> nodes = nodeRepository.findAll();
        List<NodeInfo> updatedNodes = scenarioService.applyCurrentScenarioToNodes(nodes);
        
        // Save updated nodes
        nodeRepository.saveAll(updatedNodes);
        
        // Broadcast updates via WebSocket
        // Send individual updates to maintain compatibility with existing UI listeners
        List<NodeDTO> nodeDTOs = updatedNodes.stream()
                .map(NodeDTO::fromEntity)
                .collect(Collectors.toList());
        
        // Send updates synchronously to prevent race conditions in UI
        for (NodeDTO nodeDTO : nodeDTOs) {
            messagingTemplate.convertAndSend("/topic/node-status", nodeDTO);
        }
        
        logger.info("Successfully applied scenario {} to {} nodes", scenarioName, updatedNodes.size());
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("scenario", scenarioName);
        response.put("nodesAffected", updatedNodes.size());
        response.putAll(scenarioService.getScenarioInfo(scenarioName));
        
        return ResponseEntity.ok(response);
    }

    @Operation(
        summary = "Reset về kịch bản bình thường",
        description = "Reset tất cả các node về trạng thái bình thường (NORMAL)",
        responses = {
            @ApiResponse(responseCode = "200", description = "Reset thành công")
        }
    )
    @PostMapping("/scenario/reset")
    public ResponseEntity<Map<String, Object>> resetScenario() {
        return applyScenario("NORMAL");
    }
}

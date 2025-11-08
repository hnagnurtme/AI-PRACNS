package com.sagsins.core.controller;

import com.sagsins.core.model.SimulationScenario;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * REST Controller for managing simulation scenarios
 */
@RestController
@RequestMapping("/api/simulation")
@Tag(name = "Simulation", description = "Simulation scenario management APIs")
@CrossOrigin(origins = "*")
public class SimulationController {
    
    private static final Logger logger = LoggerFactory.getLogger(SimulationController.class);
    
    private SimulationScenario currentScenario = SimulationScenario.NORMAL;
    
    @GetMapping("/scenarios")
    @Operation(summary = "List all available simulation scenarios")
    public ResponseEntity<SimulationScenario[]> getScenarios() {
        return ResponseEntity.ok(SimulationScenario.values());
    }
    
    @GetMapping("/scenario/current")
    @Operation(summary = "Get current simulation scenario")
    public ResponseEntity<Map<String, Object>> getCurrentScenario() {
        Map<String, Object> response = new HashMap<>();
        response.put("scenario", currentScenario.name());
        response.put("displayName", currentScenario.getDisplayName());
        response.put("description", currentScenario.getDescription());
        return ResponseEntity.ok(response);
    }
    
    @PostMapping("/scenario/{scenarioName}")
    @Operation(summary = "Set simulation scenario")
    public ResponseEntity<Map<String, Object>> setScenario(@PathVariable String scenarioName) {
        try {
            SimulationScenario newScenario = SimulationScenario.valueOf(scenarioName.toUpperCase());
            SimulationScenario oldScenario = currentScenario;
            currentScenario = newScenario;
            
            logger.info("🎬 Simulation scenario changed: {} → {}", 
                    oldScenario.getDisplayName(), newScenario.getDisplayName());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("previousScenario", oldScenario.name());
            response.put("currentScenario", currentScenario.name());
            response.put("message", "Scenario changed to: " + currentScenario.getDisplayName());
            
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            logger.error("Invalid scenario name: {}", scenarioName);
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("error", "Invalid scenario name: " + scenarioName);
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    @PostMapping("/scenario/reset")
    @Operation(summary = "Reset to normal scenario")
    public ResponseEntity<Map<String, Object>> resetScenario() {
        SimulationScenario oldScenario = currentScenario;
        currentScenario = SimulationScenario.NORMAL;
        
        logger.info("🔄 Simulation scenario reset: {} → NORMAL", oldScenario.getDisplayName());
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("previousScenario", oldScenario.name());
        response.put("currentScenario", "NORMAL");
        response.put("message", "Scenario reset to NORMAL");
        
        return ResponseEntity.ok(response);
    }
}

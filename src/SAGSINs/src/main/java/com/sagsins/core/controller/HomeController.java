package com.sagsins.core.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.service.INodeService;

@Controller
public class HomeController {
    private final INodeService nodeService;
    
    public HomeController(INodeService nodeService) {
        this.nodeService = nodeService;
    }

    @GetMapping("/")
    public String home(Model model) {
        List<NodeInfo> nodes = nodeService.getAllNodeIds();
        model.addAttribute("nodes", nodes);
        return "Home";
    }

    // GET /api/nodes - Get all nodes
    @GetMapping("/api/nodes")
    @ResponseBody
    public List<NodeInfo> getNodes() {
        return nodeService.getAllNodeIds();
    }

    // GET /api/nodes/{id} - Get node by ID
    @GetMapping("/api/nodes/{id}")
    @ResponseBody
    public ResponseEntity<NodeInfo> getNodeById(@PathVariable String id) {
        NodeInfo node = nodeService.getNodeById(id);
        if (node != null) {
            return ResponseEntity.ok(node);
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    // POST /api/nodes - Create new node
    @PostMapping("/api/nodes")
    @ResponseBody
    public ResponseEntity<NodeInfo> createNode(@RequestBody CreateNodeRequest request) {
        try {
            NodeInfo newNode = nodeService.createNode(request);
            return ResponseEntity.status(HttpStatus.CREATED).body(newNode);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    // PUT /api/nodes/{id} - Update node
    @PutMapping("/api/nodes/{id}")
    @ResponseBody
    public ResponseEntity<NodeInfo> updateNode(@PathVariable String id, @RequestBody UpdateNodeRequest request) {
        try {
            NodeInfo updatedNode = nodeService.updateNode(id, request);
            return ResponseEntity.ok(updatedNode);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    // DELETE /api/nodes/{id} - Delete node
    @DeleteMapping("/api/nodes/{id}")
    @ResponseBody
    public ResponseEntity<Void> deleteNode(@PathVariable String id) {
        try {
            nodeService.deleteNode(id);
            return ResponseEntity.noContent().build();
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    // POST /api/nodes/add - Legacy endpoint for backward compatibility
    @PostMapping("/api/nodes/add")
    @ResponseBody
    public NodeInfo addNode() {
        return nodeService.addNode();
    }
}

package com.sagsins.core.controller;


import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.service.INodeService;
import org.springframework.web.bind.annotation.RequestParam;




@RestController
@RequestMapping("/api/v1")
public class NodeController {
    private final INodeService nodeService;

    public NodeController(INodeService nodeService) {
        this.nodeService = nodeService;
    }

    @GetMapping("/nodes")
    public ResponseEntity<List<NodeDTO>> getAllNodes() {
        return new ResponseEntity<>(nodeService.getAllNodes(),HttpStatus.OK);
    }

    @PostMapping("/nodes")
    public ResponseEntity<NodeDTO> createNode(@RequestBody CreateNodeRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(nodeService.createNode(request));
    }

    @PatchMapping("/nodes/{nodeId}")
    public ResponseEntity<NodeDTO> updateNode(@PathVariable String nodeId, @RequestBody UpdateNodeRequest request) {
        return nodeService.updateNode(nodeId, request)
                .map(node -> ResponseEntity.ok(node))
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/nodes/{nodeId}")
    public ResponseEntity<Void> deleteNode(@PathVariable String nodeId) {
        boolean deleted = nodeService.deleteNode(nodeId);
        if (deleted) {
            return ResponseEntity.noContent().build();
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    @PostMapping("/nodes/run/{nodeId}")
    public ResponseEntity<Void> runNodeProcess(@PathVariable String nodeId) {
        boolean started = nodeService.runNodeProcess(nodeId);
        if (started) {
            return ResponseEntity.ok().build();
        } else {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}

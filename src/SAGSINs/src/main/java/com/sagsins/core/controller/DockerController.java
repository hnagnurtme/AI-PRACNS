package com.sagsins.core.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.DTOs.response.DockerResposne;
import com.sagsins.core.service.IDockerService;

@RestController
@RequestMapping("/api/v1/docker")
public class DockerController {
    private final IDockerService dockerService;

    public DockerController(IDockerService dockerService) {
        this.dockerService = dockerService;
    }

    @GetMapping("/allLinks")
    public ResponseEntity<List<DockerResposne>> getAllEntitys( @RequestParam boolean  isRunning){
        List<DockerResposne> containers = dockerService.getAllContainers(isRunning);
        return ResponseEntity.ok(containers);
    }
}

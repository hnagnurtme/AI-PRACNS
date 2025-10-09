package com.sagsins.core.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.DTOs.response.HealthResposne;

@RestController
public class CoreController {
    @GetMapping("/health")
    public ResponseEntity<HealthResposne> checkHealth() {
        return new ResponseEntity<> (new HealthResposne("OK", "Server is running"), HttpStatus.OK);
    }
}

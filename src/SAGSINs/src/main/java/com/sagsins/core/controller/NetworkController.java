package com.sagsins.core.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.model.IPAddress;
import com.sagsins.core.service.IIPScanerService;
import com.sagsins.core.service.INetworkService;
import org.springframework.web.bind.annotation.RequestParam;


@RestController
@RequestMapping("/api/v1")
public class NetworkController {
    private final INetworkService networkService;
    private final IIPScanerService ipScanerService;

    public NetworkController(INetworkService networkService , IIPScanerService ipScanerService) {
        this.networkService = networkService;
        this.ipScanerService = ipScanerService;
    }

    @GetMapping("/networks")
    public ResponseEntity<List<IPAddress>> getAllIPs() {
        return ResponseEntity.ok(networkService.getAllIPs());
    }

    @GetMapping("/networks/scan")
    public ResponseEntity<List<String>> getAvailableIps(
            @RequestParam(defaultValue = "10") int maxResults) {
        List<String> availableIps = ipScanerService.getAvailableIps(maxResults);
        return ResponseEntity.ok(availableIps);
    }
}

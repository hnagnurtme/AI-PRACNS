package com.sagsins.core.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody; 
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.DTOs.request.TwoPacket;
import com.sagsins.core.model.Packet;

@RestController
@RequestMapping("/api/v1")
public class PacketController {

    private final SimpMessagingTemplate messagingTemplate;

    private static final Logger logger = LoggerFactory.getLogger(PacketController.class);

    public PacketController(SimpMessagingTemplate messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }
    
    @PostMapping("/packets")
    public ResponseEntity<Packet> sendSinglePacket(@RequestBody Packet packet) {
        messagingTemplate.convertAndSend("/topic/packets", packet);
        logger.info("Sent packet: {}", packet);
        return ResponseEntity.ok(packet);
    }

    @PostMapping("/packets/double")
    public ResponseEntity<TwoPacket> sendDoublePacket(@RequestBody TwoPacket packet) {
        messagingTemplate.convertAndSend("/topic/packets", packet);
        logger.info("Sent packet twice: {}", packet);
        return ResponseEntity.ok(packet);   
    }
}

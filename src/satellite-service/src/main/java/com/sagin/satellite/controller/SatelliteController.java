package com.sagin.satellite.controller;

import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.BufferManager;
import com.sagin.satellite.service.TcpSender;

public class SatelliteController {

    private final BufferManager bufferManager;
    private final TcpSender tcpSender;

    public SatelliteController(BufferManager bufferManager, TcpSender tcpSender) {
        this.bufferManager = bufferManager;
        this.tcpSender = tcpSender;
    }

    public void receivePacket(Packet packet) {
        bufferManager.add(packet);
    }

    public void forwardPacket(String host, int port) throws Exception {
        Packet packet = bufferManager.poll();
        if (packet != null) {
            tcpSender.send(packet);
            System.out.println("Forwarded packet " + packet.getId() + " to " + host + ":" + port);
        }
    }
}
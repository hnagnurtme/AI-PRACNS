package com.sagin.satellite.controller;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sagin.satellite.common.SatelliteException;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.ISatelliteService;

public class SatelliteController extends BaseController {
    private final ISatelliteService satelliteService;

    private final Logger logger = LoggerFactory.getLogger(SatelliteController.class);

    public SatelliteController(ISatelliteService satelliteService ) {
        this.satelliteService = satelliteService;
    }

    public void receivePacket(Packet packet) throws SatelliteException.InvalidPacketException {
        if (packet == null || !packet.isAlive()) {
            logger.error("Received null or dead packet");
            throw new SatelliteException.InvalidPacketException("Packet is null or dead");
        }
        try {
            satelliteService.recievePacket(packet);
        }
        catch (Exception e) {
            logger.error("Error processing packet {}: {}", packet.getPacketId(), e.getMessage());
            throw new SatelliteException.InvalidPacketException(e.getMessage());
        }
    }
}

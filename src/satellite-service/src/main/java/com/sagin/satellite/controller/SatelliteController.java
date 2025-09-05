package com.sagin.satellite.controller;

import java.util.logging.Logger;

import com.sagin.satellite.common.SatelliteException;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.ISatelliteService;

public class SatelliteController extends BaseController {
    private final ISatelliteService satelliteService;

    private final Logger logger = Logger.getLogger(SatelliteController.class.getName());

    public SatelliteController(ISatelliteService satelliteService ) {
        this.satelliteService = satelliteService;
    }

    public void receivePacket(Packet packet) throws SatelliteException.InvalidPacketException {
        if (packet == null || !packet.isAlive()) {
            logger.warning("Received invalid or dead packet");
            throw new SatelliteException.InvalidPacketException("Packet is null or dead");
        }
        try {
            satelliteService.recievePacket(packet);
        }
        catch (Exception e) {
            logger.severe("Error adding packet to buffer: " + e.getMessage());
            throw new SatelliteException.InvalidPacketException(e.getMessage());
        }
    }
}

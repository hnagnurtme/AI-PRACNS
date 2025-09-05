package com.sagin.satellite.service;

import com.sagin.satellite.model.Packet;

public interface ISatelliteService {
    void recievePacket(Packet packet) throws Exception;
}

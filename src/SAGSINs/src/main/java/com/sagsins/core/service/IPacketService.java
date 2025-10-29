package com.sagsins.core.service;

import java.util.List;

import com.sagsins.core.model.Packet;

public interface IPacketService {
    Packet getPacketById(String packetId);


    List<Packet> getPacketByRoutingType(boolean isRL);
}

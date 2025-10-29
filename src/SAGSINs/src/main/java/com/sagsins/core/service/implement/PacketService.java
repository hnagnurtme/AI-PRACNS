package com.sagsins.core.service.implement;

import java.util.List;

import org.springframework.stereotype.Service;

import com.sagsins.core.model.Packet;
import com.sagsins.core.repository.IPacketRepository;
import com.sagsins.core.service.IPacketService;

@Service
public class PacketService implements IPacketService {

    private final IPacketRepository packetRepository;

    public PacketService(IPacketRepository packetRepository) {
        this.packetRepository = packetRepository;
    }
    

    @Override 
    public Packet getPacketById(String packetId) {
        return packetRepository.findById(packetId).orElse(null);
    }


    @Override
    public List<Packet> getPacketByRoutingType(boolean isRL) {
        return packetRepository.findAll().stream()
                .filter(packet -> packet.isUseRL() == isRL)
                .toList();
    }
}

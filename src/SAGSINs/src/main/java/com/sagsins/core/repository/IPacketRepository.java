package com.sagsins.core.repository;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;
import com.sagsins.core.model.Packet;

@Repository
public interface IPacketRepository extends MongoRepository<Packet, String> {
    
}

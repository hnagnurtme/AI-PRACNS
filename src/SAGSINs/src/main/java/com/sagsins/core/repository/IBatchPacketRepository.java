package com.sagsins.core.repository;

import org.bson.types.ObjectId;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import com.sagsins.core.DTOs.request.BatchPacket;

@Repository
public interface IBatchPacketRepository extends MongoRepository<BatchPacket, ObjectId> {
    
}

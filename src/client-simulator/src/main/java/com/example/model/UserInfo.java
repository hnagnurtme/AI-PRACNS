package com.example.model;

import org.bson.BsonType;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.codecs.pojo.annotations.BsonProperty;
import org.bson.codecs.pojo.annotations.BsonRepresentation;

import lombok.Data;

@Data
public class UserInfo {
    @BsonId
    @BsonRepresentation(BsonType.OBJECT_ID)
    private String id;

    @BsonProperty("userId")
    private String userId;
    private String userName;
    private String ipAddress;
    private int port;
    private String cityName;
}
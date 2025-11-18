package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum ServiceType {
    @JsonProperty("VIDEO_STREAMING")
    VIDEO_STREAM,

    @JsonProperty("AUDIO_CALL")
    AUDIO_CALL,

    @JsonProperty("IMAGE_TRANSFER")
    IMAGE_TRANSFER,

    @JsonProperty("TEXT_MESSAGE")
    TEXT_MESSAGE,

    @JsonProperty("FILE_TRANSFER")
    FILE_TRANSFER,

    // Backward compatibility - also accept VIDEO_STREAMING as enum value
    VIDEO_STREAMING
}
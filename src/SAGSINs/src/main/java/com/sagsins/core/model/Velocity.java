package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"speed", "moving"}, ignoreUnknown = true)
public record Velocity(double velocityX, double velocityY, double velocityZ) {

    @JsonProperty(value = "speed", access = JsonProperty.Access.READ_ONLY)
    public double getSpeed() {
        return Math.sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ);
    }

    @JsonIgnore
    public boolean isMoving() {
        return getSpeed() > 1e-3;
    }
}
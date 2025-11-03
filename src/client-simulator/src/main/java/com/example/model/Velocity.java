package com.example.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@JsonIgnoreProperties(value = {"speed", "moving"}, ignoreUnknown = true)
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Velocity {
    private double velocityX;
    private double velocityY;
    private double velocityZ;

    @JsonProperty(value = "speed", access = JsonProperty.Access.READ_ONLY)
    public double getSpeed() {
        return Math.sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ);
    }

    @JsonIgnore
    public boolean isMoving() {
        return getSpeed() > 1e-3;
    }
}
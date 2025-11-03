package com.example.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@ToString
@JsonIgnoreProperties(value = {"distanceTo","directionTo"}, ignoreUnknown = true)
public class Position {

    private double latitude;   
    private double longitude;  
    private double altitude;   

    public Position(double lat, double lon, double alt) {
        this.latitude = lat;
        this.longitude = lon;
        this.altitude = alt;
    }

    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    /** Khoảng cách Euclidean 3D giữa 2 điểm (xấp xỉ). */
    public double distanceTo(Position other) {
        if (other == null) return Double.MAX_VALUE;
        double dx = this.latitude - other.latitude;
        double dy = this.longitude - other.longitude;
        double dz = this.altitude - other.altitude;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    public double[] directionTo(Position other) {
        if (other == null) return new double[]{0, 0, 0};
        return new double[]{
                other.latitude - this.latitude,
                other.longitude - this.longitude,
                other.altitude - this.altitude
        };
    }
}
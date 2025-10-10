package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@ToString
@JsonIgnoreProperties(value = {"distanceTo","directionTo"}, ignoreUnknown = true)
public class Geo3D {

    private double latitude;   
    private double longitude;  
    private double altitude;   

    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    /** Khoảng cách Euclidean 3D giữa 2 điểm (xấp xỉ). */
    public double distanceTo(Geo3D other) {
        if (other == null) return Double.MAX_VALUE;
        double dx = this.latitude - other.latitude;
        double dy = this.longitude - other.longitude;
        double dz = this.altitude - other.altitude;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    /** Vector hướng từ điểm này tới điểm khác (trả về {dx, dy, dz}). */
    public double[] directionTo(Geo3D other) {
        if (other == null) return new double[]{0, 0, 0};
        return new double[]{
                other.latitude - this.latitude,
                other.longitude - this.longitude,
                other.altitude - this.altitude
        };
    }
}
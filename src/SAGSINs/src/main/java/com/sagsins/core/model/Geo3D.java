package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.JsonIgnore;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@ToString
public class Geo3D {

    private double latitude;   
    private double longitude;  
    private double altitude;   

    @JsonIgnore
    /** Khoảng cách Euclidean 3D giữa 2 điểm (xấp xỉ). */
    public double distanceTo(Geo3D other) {
        if (other == null) return Double.MAX_VALUE;
        double dx = this.latitude - other.latitude;
        double dy = this.longitude - other.longitude;
        double dz = this.altitude - other.altitude;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /** Vector hướng từ điểm này tới điểm khác (trả về {dx, dy, dz}). */
    @JsonIgnore
    public double[] directionTo(Geo3D other) {
        if (other == null) return new double[]{0, 0, 0};
        return new double[]{
                other.latitude - this.latitude,
                other.longitude - this.longitude,
                other.altitude - this.altitude
        };
    }
}
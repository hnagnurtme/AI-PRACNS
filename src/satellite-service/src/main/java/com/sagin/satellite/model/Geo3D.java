package com.sagin.satellite.model;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@ToString
public class Geo3D {

    private double latitude;   // vĩ độ (degree)
    private double longitude;  // kinh độ (degree)
    private double altitude;   // độ cao (km)

    /**
     * Khoảng cách Euclidean 3D giữa 2 điểm
     * Đây là approximation, không phải khoảng cách theo địa cầu
     */
    public double distanceTo(Geo3D other) {
        if (other == null) return Double.MAX_VALUE;
        double dx = this.latitude - other.latitude;
        double dy = this.longitude - other.longitude;
        double dz = this.altitude - other.altitude;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Vector hướng từ điểm này tới điểm khác
     * Trả về array {dx, dy, dz}
     */
    public double[] directionTo(Geo3D other) {
        if (other == null) return new double[]{0, 0, 0};
        return new double[]{
                other.latitude - this.latitude,
                other.longitude - this.longitude,
                other.altitude - this.altitude
        };
    }

    /**
     * Cập nhật vị trí mới
     */
    public void update(double latitude, double longitude, double altitude) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.altitude = altitude;
    }
}

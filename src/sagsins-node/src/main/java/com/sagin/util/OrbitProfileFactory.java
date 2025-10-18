package com.sagin.util;

import com.sagin.model.Orbit;

public final class OrbitProfileFactory {

    // 🌍 Hằng số vật lý
    private static final double EARTH_RADIUS_KM = 6371.0;       // km
    private static final double EARTH_MU = 398600.4418;         // km^3/s^2 (tham số hấp dẫn tiêu chuẩn)

    private OrbitProfileFactory() {
        // chặn tạo instance
    }

    /**
     * 🛰 Tính độ cao (altitude) hiện tại của vệ tinh từ tham số quỹ đạo.
     *
     * @param orbit Thông tin quỹ đạo (semi-major axis, eccentricity, true anomaly, ...)
     * @return độ cao tính từ bề mặt Trái đất (km)
     */
    public static double computeAltitudeKm(Orbit orbit) {
        if (orbit == null) return 0.0;

        double a = orbit.semiMajorAxisKm();      // bán trục lớn (km)
        double e = orbit.eccentricity();         // độ lệch tâm
        double ν = Math.toRadians(orbit.trueAnomalyDeg()); // true anomaly (radians)

        // r = a(1 - e^2) / (1 + e*cos(ν))
        double r = a * (1 - e * e) / (1 + e * Math.cos(ν));
        return Math.max(0, r - EARTH_RADIUS_KM);
    }

    /**
     * ⚡ Tính vận tốc quỹ đạo (km/s) tại vị trí hiện tại.
     */
    public static double computeOrbitalVelocity(Orbit orbit) {
        if (orbit == null) return 0.0;

        double a = orbit.semiMajorAxisKm();
        double e = orbit.eccentricity();
        double ν = Math.toRadians(orbit.trueAnomalyDeg());
        double r = a * (1 - e * e) / (1 + e * Math.cos(ν));

        // công thức vis-viva: v = sqrt(μ * (2/r - 1/a))
        return Math.sqrt(EARTH_MU * (2 / r - 1 / a));
    }

    /**
     * ⏱ Chu kỳ quỹ đạo (s)
     */
    public static double computeOrbitalPeriod(Orbit orbit) {
        if (orbit == null) return 0.0;

        double a = orbit.semiMajorAxisKm();
        double T = 2 * Math.PI * Math.sqrt(Math.pow(a, 3) / EARTH_MU);
        return T;
    }

    /**
     * 📈 Xác định loại quỹ đạo dựa trên độ cao trung bình
     */
    public static String classifyOrbit(double altitudeKm) {
        if (altitudeKm < 2000) return "LEO";
        if (altitudeKm < 20000) return "MEO";
        return "GEO";
    }

    /**
     * 🧮 Gói tiện ích: Trả về chuỗi mô tả nhanh
     */
    public static String describeOrbit(Orbit orbit) {
        double altitude = computeAltitudeKm(orbit);
        double velocity = computeOrbitalVelocity(orbit);
        double period = computeOrbitalPeriod(orbit) / 3600.0; // giờ

        return String.format(
            "Altitude: %.2f km | Velocity: %.2f km/s | Period: %.2f h",
            altitude, velocity, period
        );
    }
}

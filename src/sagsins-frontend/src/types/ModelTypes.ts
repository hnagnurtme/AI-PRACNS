/** Định nghĩa vị trí 3D cơ bản cho Node */
export interface Geo3D {
    latitude: number;
    longitude: number;
    altitude: number; // Km
}

/** Định nghĩa thông số quỹ đạo (chủ yếu cho vệ tinh) */
export interface Orbit {
    semiMajorAxisKm: number;
    eccentricity: number;
    inclinationDeg: number;
    raanDeg: number;
    argumentOfPerigeeDeg: number;
    trueAnomalyDeg: number;
}

/** Định nghĩa vector vận tốc 3D */
export interface Velocity {
    velocityX: number; // Km/s
    velocityY: number;
    velocityZ: number;
}
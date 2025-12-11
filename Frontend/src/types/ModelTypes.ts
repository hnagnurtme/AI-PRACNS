export interface Orbit {
    semiMajorAxisKm: number;
    eccentricity: number;
    inclinationDeg: number;
    raanDeg: number;
    argumentOfPerigeeDeg: number;
    trueAnomalyDeg: number;
}
export interface Position {
    latitude: number;
    longitude: number;
    altitude: number;
}

export interface Velocity {
    velocityX: number;
    velocityY: number;
    velocityZ: number;
}

export interface Communication {
    frequencyGHz: number;
    bandwidthMHz: number;
    transmitPowerDbW: number;
    antennaGainDb: number;
    beamWidthDeg: number;
    maxRangeKm: number;
    minElevationDeg: number;
    protocol?: string;
    ipAddress?: string;
    port?: number;
}
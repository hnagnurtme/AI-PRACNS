package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.JsonInclude;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record Communication(
    double frequencyGHz,
    double bandwidthMHz,
    double transmitPowerDbW,
    double antennaGainDb,
    double beamWidthDeg,
    double maxRangeKm,
    double minElevationDeg,
    String ipAddress,
    int port,
    String protocol
) {}
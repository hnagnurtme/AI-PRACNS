package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonInclude;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
@AllArgsConstructor
@NoArgsConstructor 
public class Communication{
    double frequencyGHz;
    double bandwidthMHz;
    double transmitPowerDbW;
    double antennaGainDb;
    double beamWidthDeg;
    double maxRangeKm;
    double minElevationDeg;
    String ipAddress;
    int port;
    String protocol;
}
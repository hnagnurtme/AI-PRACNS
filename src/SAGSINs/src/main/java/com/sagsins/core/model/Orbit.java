package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(value = {"circular"}, ignoreUnknown = true)
public record Orbit(
    double semiMajorAxisKm,
    double eccentricity,
    double inclinationDeg,
    double raanDeg,
    double argumentOfPerigeeDeg,
    double trueAnomalyDeg
) {
    @JsonProperty(value = "circular", access = JsonProperty.Access.READ_ONLY)
    public boolean isCircular() {
        return eccentricity < 1e-6;
    }
}
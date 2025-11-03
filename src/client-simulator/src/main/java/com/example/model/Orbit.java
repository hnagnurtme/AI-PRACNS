package com.example.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(value = {"circular"}, ignoreUnknown = true)
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Orbit {
    private double semiMajorAxisKm;
    private double eccentricity;
    private double inclinationDeg;
    private double raanDeg;
    private double argumentOfPerigeeDeg;
    private double trueAnomalyDeg;

    @JsonProperty(value = "circular", access = JsonProperty.Access.READ_ONLY)
    public boolean isCircular() {
        return eccentricity < 1e-6;
    }
}
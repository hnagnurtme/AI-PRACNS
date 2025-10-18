package com.sagsins.core.model;

public enum WeatherCondition {
    CLEAR(0.0),
    LIGHT_RAIN(0.5),
    RAIN(1.5),
    SNOW(1.0),
    STORM(5.0),
    SEVERE_STORM(10.0);

    private final double typicalAttenuationDb;

    WeatherCondition(double typicalAttenuationDb) {
        this.typicalAttenuationDb = typicalAttenuationDb;
    }

    public double getTypicalAttenuationDb() {
        return typicalAttenuationDb;
    }
}
package com.sagsins.core.model;


public enum WeatherCondition {
    CLEAR(0.0),             // Trời quang, suy hao tối thiểu
    LIGHT_RAIN(0.5),        // Mưa nhẹ, suy hao thấp (0.5 dB)
    RAIN(1.5),              // Mưa vừa, suy hao trung bình (1.5 dB)
    SNOW(1.0),              // Tuyết, suy hao trung bình (1.0 dB)
    STORM(5.0),             // Bão, suy hao đáng kể (5.0 dB)
    SEVERE_STORM(10.0);     // Bão lớn/Băng giá, suy hao nghiêm trọng (10.0 dB)

    private final double typicalAttenuationDb;

    WeatherCondition(double typicalAttenuationDb) {
        this.typicalAttenuationDb = typicalAttenuationDb;
    }

    public double getTypicalAttenuationDb() {
        return typicalAttenuationDb;
    }
}
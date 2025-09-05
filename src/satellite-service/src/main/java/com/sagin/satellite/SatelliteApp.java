package com.sagin.satellite;

import com.sagin.satellite.config.FireStoreConfiguration;

public class SatelliteApp {
    public static void main(String[] args) throws Exception {
        FireStoreConfiguration.init();

        
        FireStoreConfiguration.shutdown();
    }
}
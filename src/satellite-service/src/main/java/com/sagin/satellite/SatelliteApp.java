package com.sagin.satellite;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sagin.satellite.config.ApplicationConfiguration;

public class SatelliteApp {
    private static final Logger logger = LoggerFactory.getLogger(SatelliteApp.class);
    public static void main(String[] args) {
        ApplicationConfiguration.init();
        ApplicationConfiguration.initServer(6000);
        logger.info("Satellite Service is running on port 6000");
    }
}

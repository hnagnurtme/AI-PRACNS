package com.sagin.satellite;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sagin.satellite.config.ApplicationConfiguration;

public class SatelliteApp {
    private static final Logger logger = LoggerFactory.getLogger(SatelliteApp.class);
    public static void main(String[] args) {
        ApplicationConfiguration.init();
        String portEnv = System.getenv("SERVER_PORT");
        int port = portEnv != null ? Integer.parseInt(portEnv) : 3000;
        ApplicationConfiguration.initServer(port);
        logger.info("Satellite Service started on port {}", port);
    }
}

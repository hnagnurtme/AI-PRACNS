package com.sagin.satellite;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sagin.satellite.config.ApplicationConfiguration;
import com.sagin.satellite.view.SatelliteWebView;

public class SatelliteApp {
    private static final Logger logger = LoggerFactory.getLogger(SatelliteApp.class);
    public static void main(String[] args) {
        ApplicationConfiguration.init();
        ApplicationConfiguration.initServer(6000);
        try {
            SatelliteWebView.startServer(8080);
        } catch (Exception e) {
            logger.error("Failed to initialize Firebase: {}", e.getMessage());
            return;
        }
        logger.info("Satellite Service is running on port 6000");
    }
}

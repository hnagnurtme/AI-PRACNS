package com.sagin.satellite.config;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sagin.satellite.common.SatelliteException;
import com.sagin.satellite.controller.SatelliteController;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.IBufferManager;
import com.sagin.satellite.service.ISatelliteService;
import com.sagin.satellite.service.implement.BufferManager;
import com.sagin.satellite.service.implement.SatelliteService;
import com.sagin.satellite.service.implement.TcpSender;

public class ApplicationConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(ApplicationConfiguration.class);

    private static SatelliteController satelliteController;

    public static void init() {
        try {
            // 1️⃣ Init Firebase 
            FireBaseConfiguration.init();

            TcpSender tcpSender = new TcpSender();

            // 3️⃣ Init BufferManager
            int bufferCapacity = 1000;
            int flushIntervalMs = 50;
            int maxRetry = 3;
            IBufferManager bufferManager = new BufferManager(bufferCapacity, tcpSender, flushIntervalMs, maxRetry, 4);
            // 4️⃣ Init SatelliteService
            ISatelliteService satelliteService = new SatelliteService(bufferManager);

            // 5️⃣ Init Controller
            satelliteController = new SatelliteController(satelliteService);

            logger.info("ApplicationConfiguration initialized successfully");
        } catch (Exception e) {
            logger.error("Failed to initialize application configuration", e);
            throw new RuntimeException("Failed to initialize configuration", e);
        }
    }

    public static void initServer(int port) {
        if (satelliteController == null) {
            throw new IllegalStateException("SatelliteController not initialized. Call init() first.");
        }

        new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                logger.info("Satellite Server listening on port {}", port);
                ObjectMapper mapper = new ObjectMapper();

                while (true) {
                    Socket clientSocket = serverSocket.accept();
                    handleClient(clientSocket, mapper);
                }
            } catch (Exception e) {
                logger.error("Server error: ", e);
            }
        }).start();
    }

    private static void handleClient(Socket clientSocket, ObjectMapper mapper) {
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                 OutputStream out = clientSocket.getOutputStream()) {

                String json = reader.readLine();
                if (json != null && !json.isEmpty()) {
                    Packet packet = mapper.readValue(json, Packet.class);

                    try {
                        satelliteController.receivePacket(packet);
                        out.write("SUCCESS\n".getBytes());
                        out.flush();
                    } catch (SatelliteException.InvalidPacketException e) {
                        out.write(("ERROR: " + e.getMessage() + "\n").getBytes());
                        out.flush();
                    }
                }
            } catch (Exception ex) {
                logger.error("Error handling client connection: ", ex);
            } finally {
                try { clientSocket.close(); } catch (Exception ignored) {}
            }
        }).start();
    }

    public static SatelliteController getSatelliteController() {
        return satelliteController;
    }
}

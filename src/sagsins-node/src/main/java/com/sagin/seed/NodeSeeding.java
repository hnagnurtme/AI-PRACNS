package com.sagin.seed;

import com.sagin.model.*;
import com.sagin.repository.MongoNodeRepository;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Seeding 15 nodes in Asia (Ground Stations + LEO/MEO/GEO Satellites)
 * with TCP, 127.0.0.1, Velocity record, and realistic Orbit for satellites.
 */
public class NodeSeeding {

    public static void main(String[] args) {
        List<NodeInfo> nodes = new ArrayList<>();

        // -------------------------
        // 1. Ground Stations
        // -------------------------
        nodes.add(createTCPGroundStation("node-01", "N-TOKYO", "Tokyo Ground Station", 35.6895, 139.6917, 5001));
        nodes.add(createTCPGroundStation("node-02", "N-HANOI", "Hanoi Ground Station", 21.0285, 105.8542, 5002));
        nodes.add(createTCPGroundStation("node-03", "N-BANGKOK", "Bangkok Ground Station", 13.7563, 100.5018, 5003));
        nodes.add(createTCPGroundStation("node-04", "N-DELHI", "Delhi Ground Station", 28.6139, 77.2090, 5004));
        nodes.add(createTCPGroundStation("node-05", "N-SINGAPORE", "Singapore Ground Station", 1.3521, 103.8198, 5005));

        // -------------------------
        // 2. LEO Satellites (500 km, 7.61 km/s)
        // -------------------------
        nodes.add(createTCPSatellite("node-06", "SAT-LEO-1", "LEO Sat 1", NodeType.LEO_SATELLITE, 30.0, 120.0, 500, 7.61, 6001));
        nodes.add(createTCPSatellite("node-07", "SAT-LEO-2", "LEO Sat 2", NodeType.LEO_SATELLITE, 10.0, 100.0, 500, 7.61, 6002));
        nodes.add(createTCPSatellite("node-08", "SAT-LEO-3", "LEO Sat 3", NodeType.LEO_SATELLITE, 40.0, 90.0, 500, 7.61, 6003));

        // -------------------------
        // 3. GEO Satellites (35786 km, 3.07 km/s)
        // -------------------------
        nodes.add(createTCPSatellite("node-09", "SAT-GEO-1", "GEO Sat 1", NodeType.GEO_SATELLITE, 0.0, 105.0, 35786, 3.07, 6004));
        nodes.add(createTCPSatellite("node-10", "SAT-GEO-2", "GEO Sat 2", NodeType.GEO_SATELLITE, 10.0, 110.0, 35786, 3.07, 6005));

        // -------------------------
        // 4. MEO Satellites (10000 km, 3.87 km/s approx.)
        // -------------------------
        nodes.add(createTCPSatellite("node-11", "SAT-MEO-1", "MEO Sat 1", NodeType.MEO_SATELLITE, 20.0, 100.0, 10000, 3.87, 6006));
        nodes.add(createTCPSatellite("node-12", "SAT-MEO-2", "MEO Sat 2", NodeType.MEO_SATELLITE, 25.0, 95.0, 10000, 3.87, 6007));
        nodes.add(createTCPSatellite("node-13", "SAT-MEO-3", "MEO Sat 3", NodeType.MEO_SATELLITE, 15.0, 105.0, 10000, 3.87, 6008));
        nodes.add(createTCPSatellite("node-14", "SAT-MEO-4", "MEO Sat 4", NodeType.MEO_SATELLITE, 30.0, 110.0, 10000, 3.87, 6009));
        nodes.add(createTCPSatellite("node-15", "SAT-MEO-5", "MEO Sat 5", NodeType.MEO_SATELLITE, 10.0, 115.0, 10000, 3.87, 6010));

        // -------------------------
        // Insert into MongoDB
        // -------------------------
        try (MongoNodeRepository repo = new MongoNodeRepository()) {
            repo.bulkUpdateNodes(nodes);
            System.out.println("Seeded 15 Asia nodes successfully with TCP, 127.0.0.1, Velocity, and Orbit.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // -------------------------
    // Helper methods
    // -------------------------
    private static NodeInfo createTCPGroundStation(String id, String nodeId, String name, double lat, double lon, int port) {
        Velocity velStation = new Velocity(0.0, 0.0, 0.0);

        return new NodeInfo(
                null,
                nodeId,
                name,
                NodeType.GROUND_STATION,
                new Position(lat, lon, 0),
                null, // Ground stations không cần Orbit
                velStation,
                new Communication(2.2, 100, 20, 30, 60, 2000, 10, "127.0.0.1", port, "TCP"),
                true,
                100,
                1,
                0,
                0,
                100,
                0,
                WeatherCondition.CLEAR,
                Instant.now(),
                "127.0.0.1",
                port
        );
    }

    private static NodeInfo createTCPSatellite(String id, String nodeId, String name, NodeType type,
                                            double lat, double lon, double altitude, double speedX, int port) {
        Velocity velSatellite = new Velocity(speedX, 0.0, 0.0);
        Orbit orbit;

        switch (type) {
            case LEO_SATELLITE -> orbit = new Orbit(
                    6378 + 500, 0.001, 97.5, 0.0, 0.0, 0.0
            );
            case MEO_SATELLITE -> orbit = new Orbit(
                    6378 + 10000, 0.001, 55.0, 0.0, 0.0, 0.0
            );
            default -> orbit = new Orbit(
                    6378 + 35786, 0.0, 0.0, 0.0, 0.0, 0.0
            ); // GEO
        }

        return new NodeInfo(
                null,
                nodeId,
                name,
                type,
                new Position(lat, lon, altitude),
                orbit,
                velSatellite,
                new Communication(8.2, 500, 40, 40, 10, 3000, 20, "127.0.0.1", port, "TCP"),
                true,
                100,
                type == NodeType.LEO_SATELLITE ? 2 : type == NodeType.MEO_SATELLITE ? 3 : 5,
                0,
                0,
                type == NodeType.LEO_SATELLITE ? 200 : type == NodeType.MEO_SATELLITE ? 300 : 500,
                0,
                WeatherCondition.CLEAR,
                Instant.now(),
                "127.0.0.1",
                port
        );
    }
}

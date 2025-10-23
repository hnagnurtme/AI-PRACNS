package com.sagin.seed;

import com.sagin.model.*;
import com.sagin.repository.MongoNodeRepository;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Seeding 30 global nodes (8 Ground Stations + 22 Satellites)
 * with realistic coordinates, altitudes, velocities, and TCP configuration.
 */
public class NodeSeeding{

    public static void main(String[] args) {
        List<NodeInfo> nodes = new ArrayList<>();

        // -------------------------
        // 1. Ground Stations (8) - distributed globally
        // -------------------------
        nodes.add(createTCPGroundStation("node-01", "N-TOKYO", "Tokyo Ground Station", 35.6895, 139.6917, 5001));
        nodes.add(createTCPGroundStation("node-02", "N-HANOI", "Hanoi Ground Station", 21.0285, 105.8542, 5002));
        nodes.add(createTCPGroundStation("node-03", "N-PARIS", "Paris Ground Station", 48.8566, 2.3522, 5003));
        nodes.add(createTCPGroundStation("node-04", "N-CAPE", "Cape Town Ground Station", -33.9249, 18.4241, 5004));
        nodes.add(createTCPGroundStation("node-05", "N-NEWYORK", "New York Ground Station", 40.7128, -74.0060, 5005));
        nodes.add(createTCPGroundStation("node-06", "N-SYDNEY", "Sydney Ground Station", -33.8688, 151.2093, 5006));
        nodes.add(createTCPGroundStation("node-07", "N-RIO", "Rio de Janeiro Ground Station", -22.9068, -43.1729, 5007));
        nodes.add(createTCPGroundStation("node-08", "N-ANCHORAGE", "Anchorage Ground Station", 61.2181, -149.9003, 5008));

        // -------------------------
        // 2. LEO Satellites (8) - 500 km, ~7.61 km/s
        // -------------------------
        nodes.add(createTCPSatellite("node-09", "SAT-LEO-1", "LEO Sat 1", NodeType.LEO_SATELLITE, 0.0, 0.0, 500, 7.61, 6001));
        nodes.add(createTCPSatellite("node-10", "SAT-LEO-2", "LEO Sat 2", NodeType.LEO_SATELLITE, 15.0, 90.0, 500, 7.61, 6002));
        nodes.add(createTCPSatellite("node-11", "SAT-LEO-3", "LEO Sat 3", NodeType.LEO_SATELLITE, 30.0, 180.0, 500, 7.61, 6003));
        nodes.add(createTCPSatellite("node-12", "SAT-LEO-4", "LEO Sat 4", NodeType.LEO_SATELLITE, -15.0, -90.0, 500, 7.61, 6004));
        nodes.add(createTCPSatellite("node-13", "SAT-LEO-5", "LEO Sat 5", NodeType.LEO_SATELLITE, 45.0, 45.0, 500, 7.61, 6005));
        nodes.add(createTCPSatellite("node-14", "SAT-LEO-6", "LEO Sat 6", NodeType.LEO_SATELLITE, -45.0, -45.0, 500, 7.61, 6006));
        nodes.add(createTCPSatellite("node-15", "SAT-LEO-7", "LEO Sat 7", NodeType.LEO_SATELLITE, 60.0, 120.0, 500, 7.61, 6007));
        nodes.add(createTCPSatellite("node-16", "SAT-LEO-8", "LEO Sat 8", NodeType.LEO_SATELLITE, -60.0, -120.0, 500, 7.61, 6008));

        // -------------------------
        // 3. MEO Satellites (8) - 10,000 km, ~3.87 km/s
        // -------------------------
        nodes.add(createTCPSatellite("node-17", "SAT-MEO-1", "MEO Sat 1", NodeType.MEO_SATELLITE, 10.0, 0.0, 10000, 3.87, 6011));
        nodes.add(createTCPSatellite("node-18", "SAT-MEO-2", "MEO Sat 2", NodeType.MEO_SATELLITE, 20.0, 45.0, 10000, 3.87, 6012));
        nodes.add(createTCPSatellite("node-19", "SAT-MEO-3", "MEO Sat 3", NodeType.MEO_SATELLITE, 30.0, 90.0, 10000, 3.87, 6013));
        nodes.add(createTCPSatellite("node-20", "SAT-MEO-4", "MEO Sat 4", NodeType.MEO_SATELLITE, -10.0, -45.0, 10000, 3.87, 6014));
        nodes.add(createTCPSatellite("node-21", "SAT-MEO-5", "MEO Sat 5", NodeType.MEO_SATELLITE, -20.0, -90.0, 10000, 3.87, 6015));
        nodes.add(createTCPSatellite("node-22", "SAT-MEO-6", "MEO Sat 6", NodeType.MEO_SATELLITE, 45.0, 135.0, 10000, 3.87, 6016));
        nodes.add(createTCPSatellite("node-23", "SAT-MEO-7", "MEO Sat 7", NodeType.MEO_SATELLITE, -45.0, -135.0, 10000, 3.87, 6017));
        nodes.add(createTCPSatellite("node-24", "SAT-MEO-8", "MEO Sat 8", NodeType.MEO_SATELLITE, 0.0, 180.0, 10000, 3.87, 6018));

        // -------------------------
        // 4. GEO Satellites (6) - 35,786 km, ~3.07 km/s
        // -------------------------
        nodes.add(createTCPSatellite("node-25", "SAT-GEO-1", "GEO Sat 1", NodeType.GEO_SATELLITE, 0.0, 0.0, 35786, 3.07, 6021));
        nodes.add(createTCPSatellite("node-26", "SAT-GEO-2", "GEO Sat 2", NodeType.GEO_SATELLITE, 0.0, 60.0, 35786, 3.07, 6022));
        nodes.add(createTCPSatellite("node-27", "SAT-GEO-3", "GEO Sat 3", NodeType.GEO_SATELLITE, 0.0, 120.0, 35786, 3.07, 6023));
        nodes.add(createTCPSatellite("node-28", "SAT-GEO-4", "GEO Sat 4", NodeType.GEO_SATELLITE, 0.0, 180.0, 35786, 3.07, 6024));
        nodes.add(createTCPSatellite("node-29", "SAT-GEO-5", "GEO Sat 5", NodeType.GEO_SATELLITE, 0.0, -60.0, 35786, 3.07, 6025));
        nodes.add(createTCPSatellite("node-30", "SAT-GEO-6", "GEO Sat 6", NodeType.GEO_SATELLITE, 0.0, -120.0, 35786, 3.07, 6026));

        // -------------------------
        // Insert all nodes
        // -------------------------
        try (MongoNodeRepository repo = new MongoNodeRepository()) {
            repo.bulkUpdateNodes(nodes);
            System.out.println("✅ Seeded 30 global nodes successfully into MongoDB.");
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
                null,
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
            case LEO_SATELLITE -> orbit = new Orbit(6378 + 500, 0.001, 97.5, 0.0, 0.0, 0.0);
            case MEO_SATELLITE -> orbit = new Orbit(6378 + 10000, 0.001, 55.0, 0.0, 0.0, 0.0);
            default -> orbit = new Orbit(6378 + 35786, 0.0, 0.0, 0.0, 0.0, 0.0);
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

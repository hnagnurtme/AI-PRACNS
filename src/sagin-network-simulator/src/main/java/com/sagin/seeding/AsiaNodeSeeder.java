package com.sagin.seeding;

import com.sagin.model.*;
import com.sagin.repository.INodeRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class AsiaNodeSeeder {

    private static final Logger logger = LoggerFactory.getLogger(AsiaNodeSeeder.class);
    private final INodeRepository nodeRepository;
    private static final int BASE_PORT = 8080;

    public AsiaNodeSeeder(INodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    public void seedNodes(boolean forceOverwrite) {
        logger.info("Seeding Node khu vực Châu Á...");

        Map<String, NodeInfo> nodes = createNodes();

        Map<String, NodeInfo> current = nodeRepository.loadAllNodeConfigs();
        if (!current.isEmpty() && !forceOverwrite) {
            logger.warn("Database đã có {} Node, bỏ qua seeding.", current.size());
            return;
        }

        nodes.forEach((id, n) -> nodeRepository.updateNodeInfo(id, n));
        logger.info("Đã seed {} Node khu vực Châu Á.", nodes.size());
    }

    public Map<String, NodeInfo> createNodes() {
        Map<String, NodeInfo> nodes = new HashMap<>();
        long now = System.currentTimeMillis();
        int port = BASE_PORT;

        // --- GROUND STATIONS ---
        nodes.put("GS_LONDON", createGroundStation("GS_LONDON", 51.5074, -0.1278, 0.01, WeatherCondition.CLEAR, 1.0, port++));
        nodes.put("GS_TOKYO", createGroundStation("GS_TOKYO", 35.68, 139.69, 0.01, WeatherCondition.CLEAR, 1.0, port++));
        nodes.put("GS_SINGAPORE", createGroundStation("GS_SINGAPORE", 1.35, 103.82, 0.01, WeatherCondition.CLEAR, 1.0, port++));
        nodes.put("GS_BANGKOK", createGroundStation("GS_BANGKOK", 13.75, 100.50, 0.01, WeatherCondition.LIGHT_RAIN, 1.0, port++));

        // --- LEO SATELLITES ---
        nodes.put("LEO_001", createSatellite("LEO_001", NodeType.LEO_SATELLITE, 20.0, 120.0, 600, 7.5, 0.0, 0.0, 95.0, 80, 53.0, now, port++));
        nodes.put("LEO_002", createSatellite("LEO_002", NodeType.LEO_SATELLITE, 15.0, 105.0, 600, 7.5, 0.0, 0.0, 95.0, 80, 53.0, now, port++));
        nodes.put("LEO_003", createSatellite("LEO_003", NodeType.LEO_SATELLITE, 30.0, 140.0, 600, 7.5, 0.0, 0.0, 95.0, 80, 53.0, now, port++));
        nodes.put("LEO_004", createSatellite("LEO_004", NodeType.LEO_SATELLITE, 5.0, 100.0, 600, 7.5, 0.0, 0.0, 95.0, 80, 53.0, now, port++));

        // --- MEO SATELLITES ---
        nodes.put("MEO_001", createSatellite("MEO_001", NodeType.MEO_SATELLITE, 25.0, 115.0, 15000, 3.0, 0.0, 0.0, 98.0, 150, 60.0, now, port++));
        nodes.put("MEO_002", createSatellite("MEO_002", NodeType.MEO_SATELLITE, 10.0, 110.0, 15000, 3.0, 0.0, 0.0, 98.0, 150, 60.0, now, port++));

        // --- GEO SATELLITE ---
        nodes.put("GEO_001", createSatellite("GEO_001", NodeType.GEO_SATELLITE, 20.0, 100.0, 35786, 0.001, 0.0, 0.0, 99.0, 200, 0.0, now, port++));

        return nodes;
    }

    public NodeInfo createGroundStation(String id, double lat, double lon, double alt,
                                         WeatherCondition weather, double delayMs, int port) {
        NodeInfo n = new NodeInfo();
        n.setNodeId(id);
        n.setNodeType(NodeType.GROUND_STATION);
        n.setPosition(new Geo3D(lat, lon, alt));
        n.setWeather(weather);
        n.setPort(port);
        n.setOperational(true);
        n.setBatteryChargePercent(100.0);
        n.setNodeProcessingDelayMs(delayMs);
        n.setPacketLossRate(0.00001);
        n.setPacketBufferCapacity(500);
        n.setCurrentPacketCount(0);
        n.setResourceUtilization(0.0);
        n.setOrbit(null);
        n.setVelocity(null);
        n.setLastUpdated(System.currentTimeMillis());
        return n;
    }

    private NodeInfo createSatellite(String id, NodeType type, double lat, double lon, double alt,
                                     double vx, double vy, double vz, double battery,
                                     int bufferCapacity, double inclination, long timestamp, int port) {
        NodeInfo n = new NodeInfo();
        n.setNodeId(id);
        n.setNodeType(type);
        n.setPosition(new Geo3D(lat, lon, alt));
        n.setVelocity(new Velocity(vx, vy, vz));
        n.setOrbit(new Orbit(alt + 6371.0, 0.001, inclination, 0.0, 0.0, 0.0));
        n.setPort(port);
        n.setOperational(true);
        n.setBatteryChargePercent(battery);
        n.setNodeProcessingDelayMs(0.8);
        n.setPacketLossRate(0.005);
        n.setPacketBufferCapacity(bufferCapacity);
        n.setCurrentPacketCount(0);
        n.setResourceUtilization(0.1);
        n.setWeather(WeatherCondition.CLEAR);
        n.setLastUpdated(timestamp);
        return n;
    }
}

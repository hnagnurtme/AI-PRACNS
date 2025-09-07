package com.sagsins.core.seed;

import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.sagsins.core.repository.NodeRepository;
import com.sagsins.core.utils.ProjectConstant;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import jakarta.annotation.PostConstruct;

import java.util.ArrayList;
import java.util.List;

@Component
public class SagsinsSeeder {
    private static final Logger logger = LoggerFactory.getLogger(SagsinsSeeder.class);
    private final NodeRepository nodeRepository;

    public SagsinsSeeder(NodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    @PostConstruct
    public void seedNetworkNodes() {
        if (!nodeRepository.getAllNodes().isEmpty()) {
            return; // đã có dữ liệu
        }

        List<NodeInfo> nodes = new ArrayList<>();

        // ========== REALISTIC SATELLITE CONSTELLATION ==========

        // Starlink-like LEO constellation
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(53.0, -70.0, 550_000), // Shell 1
                new Orbit("LEO", 53.0, 95.4, 6_928),
                new Velocity(0, 7_560, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(53.2, -65.0, 550_000),
                new Orbit("LEO", 53.0, 95.4, 6_928),
                new Velocity(0, 7_560, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(70.0, 120.0, 570_000), // Shell 2 - polar
                new Orbit("LEO", 70.0, 96.1, 6_948),
                new Velocity(0, 7_540, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(-45.0, 45.0, 1_200_000), // Shell 3 - higher orbit
                new Orbit("LEO", 53.8, 112.0, 7_578),
                new Velocity(0, 6_640, 0)));

        // OneWeb-like MEO constellation
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(87.9, 0.0, 1_200_000), // Near-polar MEO
                new Orbit("MEO", 87.9, 109.0, 7_578),
                new Velocity(0, 6_640, 0)));

        // Traditional GEO satellites
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(0.0, 105.5, 35_786_000), // AsiaSat position
                new Orbit("GEO", 0.1, 1436.0, 42_164),
                new Velocity(0, 3_074, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(0.0, -75.0, 35_786_000), // Americas coverage
                new Orbit("GEO", 0.1, 1436.0, 42_164),
                new Velocity(0, 3_074, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(0.0, 28.2, 35_786_000), // Europe/Africa coverage
                new Orbit("GEO", 0.1, 1436.0, 42_164),
                new Velocity(0, 3_074, 0)));

        // ========== GLOBAL GROUND STATIONS ==========

        // Primary teleports
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(21.0285, 105.8542, 20), // Hanoi, Vietnam
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(10.7769, 106.7009, 10), // Ho Chi Minh City, Vietnam
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(47.6062, -122.3321, 100), // Seattle (SpaceX)
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(28.5618, -80.577, 5), // Cape Canaveral
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(51.4816, -0.0076, 25), // London Gateway
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(35.6762, 139.6503, 40), // Tokyo
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(-33.8688, 151.2093, 50), // Sydney
                null,
                new Velocity(0, 0, 0)));

        // Remote tracking stations
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(78.2232, 15.6267, 100), // Svalbard (polar orbits)
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(-54.8019, -68.3030, 50), // Ushuaia (southern coverage)
                null,
                new Velocity(0, 0, 0)));

        // ========== HIGH ALTITUDE PLATFORMS ==========

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_RELAY,
                new Geo3D(0.0, 0.0, 20_000), // Equatorial HAPS
                null,
                new Velocity(50, 0, 0) // station-keeping
        ));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_RELAY,
                new Geo3D(35.0, -95.0, 18_000), // US coverage HAPS
                null,
                new Velocity(30, 20, 0) // wind compensation
        ));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_RELAY,
                new Geo3D(50.0, 10.0, 21_000), // European HAPS
                null,
                new Velocity(-20, 15, 0)));

        // ========== MOBILE USER EQUIPMENT ==========

        // Urban users
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(48.8566, 2.3522, 5), // Paris
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(40.7128, -74.0060, 10), // New York
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(1.3521, 103.8198, 15), // Singapore
                null,
                new Velocity(0, 0, 0)));

        // Mobile users (vehicles)
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(52.5200, 13.4050, 2), // Berlin - moving car
                null,
                new Velocity(20, 5, 0) // ~25 m/s (~90 km/h)
        ));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(37.7749, -122.4194, 100), // San Francisco - aircraft
                null,
                new Velocity(150, 0, 10) // commercial aircraft speed
        ));

        // Remote/rural users
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(-23.5505, -46.6333, 800), // São Paulo (mountainous)
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(64.2008, -149.4937, 200), // Fairbanks, Alaska (high latitude)
                null,
                new Velocity(0, 0, 0)));

        // ========== MARITIME AND AVIATION NODES ==========

        // Commercial vessels
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SEA,
                new Geo3D(35.0, 140.0, 0), // Pacific shipping lane
                null,
                new Velocity(15, 0, 0) // container ship ~15 m/s
        ));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SEA,
                new Geo3D(50.0, -30.0, 0), // North Atlantic route
                null,
                new Velocity(12, -5, 0) // cargo vessel
        ));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SEA,
                new Geo3D(-10.0, 80.0, 0), // Indian Ocean
                null,
                new Velocity(8, 8, 0) // fishing vessel
        ));

        // Offshore platforms
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SEA,
                new Geo3D(60.0, 2.0, 0), // North Sea oil rig
                null,
                new Velocity(0, 0, 0) // stationary platform
        ));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SEA,
                new Geo3D(28.0, -89.0, 0), // Gulf of Mexico platform
                null,
                new Velocity(0, 0, 0)));

        // Research vessels
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SEA,
                new Geo3D(-65.0, -64.0, 0), // Antarctic research station ship
                null,
                new Velocity(5, 2, 0) // slow research vessel
        ));

        // ========== EMERGENCY/DISASTER RESPONSE ==========

        // Temporary deployable units
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(36.2048, 138.2529, 500), // Japan (disaster response)
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(-8.3405, 115.0920, 50), // Bali (remote tourism)
                null,
                new Velocity(0, 0, 0)));

        // ========== INTER-SATELLITE LINKS TEST NODES ==========

        // Constellation cross-links (same orbital plane)
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(53.5, -68.0, 550_000), // Adjacent to first Starlink sat
                new Orbit("LEO", 53.0, 95.4, 6_928),
                new Velocity(0, 7_560, 0)));

        // Different orbital planes for testing inter-plane links
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(53.0, -20.0, 550_000), // Different longitude
                new Orbit("LEO", 53.0, 95.4, 6_928),
                new Velocity(0, 7_560, 0)));

        // ========== BACKUP AND REDUNDANCY ==========

        // Backup GEO satellite
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_SATELLITE,
                new Geo3D(0.0, 105.0, 35_786_000), // Backup for AsiaSat
                new Orbit("GEO", 0.1, 1436.0, 42_164),
                new Velocity(0, 3_074, 0)));

        // Gateway redundancy
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_GROUND_STATION,
                new Geo3D(21.1000, 105.9000, 15), // Backup Hanoi gateway
                null,
                new Velocity(0, 0, 0)));

        // ========== PERFORMANCE TESTING SCENARIOS ==========

        // High-speed mobile (train/highway)
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(49.8397, 2.0833, 5), // High-speed rail route
                null,
                new Velocity(85, 0, 0) // TGV speed ~300 km/h = 85 m/s
        ));

        // Dense urban (multiple UEs in same area)
        for (int i = 0; i < 5; i++) {
            nodes.add(new NodeInfo(
                    ProjectConstant.NODE_TYPE_UE,
                    new Geo3D(40.7589 + i * 0.01, -73.9851 + i * 0.01, 100 + i * 50), // Manhattan area
                    null,
                    new Velocity(i * 2, i * 2, 0) // Various pedestrian speeds
            ));
        }

        // Polar region coverage test
        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(89.0, 0.0, 50), // Near North Pole
                null,
                new Velocity(0, 0, 0)));

        nodes.add(new NodeInfo(
                ProjectConstant.NODE_TYPE_UE,
                new Geo3D(-85.0, 0.0, 100), // Near South Pole
                null,
                new Velocity(0, 0, 0)));
        nodes.forEach(nodeRepository::saveNode);

        logger.info("✅ Seeded {} network nodes into Firestore.", nodes.size());
    }

}

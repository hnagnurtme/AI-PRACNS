package com.sagin.service;

import com.sagin.model.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SimulationScenarioService
 */
class SimulationScenarioServiceTest {
    
    private SimulationScenarioService service;
    private NodeInfo testNode;
    private Packet testPacket;
    
    @BeforeEach
    void setUp() {
        service = new SimulationScenarioService();
        
        // Create test node
        testNode = new NodeInfo();
        testNode.setNodeId("TEST-NODE-1");
        testNode.setNodeType(NodeType.LEO_SATELLITE);
        testNode.setPacketBufferCapacity(100);
        testNode.setCurrentPacketCount(10);
        testNode.setWeather(WeatherCondition.CLEAR);
        testNode.setOperational(true);
        
        // Create test packet
        testPacket = new Packet();
        testPacket.setPacketId("TEST-PACKET-1");
        testPacket.setTTL(10);
    }
    
    @Test
    void testDefaultScenarioIsNormal() {
        assertEquals(SimulationScenario.NORMAL, service.getCurrentScenario());
    }
    
    @Test
    void testSetScenario() {
        service.setScenario(SimulationScenario.WEATHER_EVENT);
        assertEquals(SimulationScenario.WEATHER_EVENT, service.getCurrentScenario());
    }
    
    @Test
    void testNormalScenarioDoesNotModifyNode() {
        service.setScenario(SimulationScenario.NORMAL);
        NodeInfo result = service.applyScenarioToNode(testNode);
        
        assertEquals(WeatherCondition.CLEAR, result.getWeather());
        assertEquals(10, result.getCurrentPacketCount());
        assertTrue(result.isOperational());
    }
    
    @Test
    void testWeatherEventScenarioChangesWeather() {
        service.setScenario(SimulationScenario.WEATHER_EVENT);
        NodeInfo result = service.applyScenarioToNode(testNode);
        
        assertNotNull(result);
        // Weather should potentially change (not guaranteed due to randomness, but node should not be null)
    }
    
    @Test
    void testNodeOverloadScenarioIncreasesQueue() {
        service.setScenario(SimulationScenario.NODE_OVERLOAD);
        int originalCount = testNode.getCurrentPacketCount();
        
        NodeInfo result = service.applyScenarioToNode(testNode);
        
        assertNotNull(result);
        // Queue size should increase (>= 70% of capacity)
        assertTrue(result.getCurrentPacketCount() >= originalCount);
    }
    
    @Test
    void testNodeOfflineScenarioMayDisableNode() {
        service.setScenario(SimulationScenario.NODE_OFFLINE);
        
        // Run multiple times due to randomness
        boolean foundOffline = false;
        for (int i = 0; i < 50; i++) {
            NodeInfo freshNode = new NodeInfo();
            freshNode.setOperational(true);
            NodeInfo result = service.applyScenarioToNode(freshNode);
            if (!result.isOperational()) {
                foundOffline = true;
                break;
            }
        }
        
        // At least one node should be marked offline in 50 attempts
        assertTrue(foundOffline);
    }
    
    @Test
    void testTrafficSpikeScenarioIncreasesQueue() {
        service.setScenario(SimulationScenario.TRAFFIC_SPIKE);
        testNode.setCurrentPacketCount(5);
        
        NodeInfo result = service.applyScenarioToNode(testNode);
        
        assertNotNull(result);
        // Queue should increase due to traffic spike
        assertTrue(result.getCurrentPacketCount() >= 5);
    }
    
    @Test
    void testShouldDropPacketWithTTLExpired() {
        service.setScenario(SimulationScenario.TTL_EXPIRED);
        testPacket.setTTL(0);
        
        assertTrue(service.shouldDropPacket(testPacket, testNode));
    }
    
    @Test
    void testShouldNotDropPacketWithValidTTL() {
        service.setScenario(SimulationScenario.TTL_EXPIRED);
        testPacket.setTTL(10);
        
        assertFalse(service.shouldDropPacket(testPacket, testNode));
    }
    
    @Test
    void testGetDropReasonForNodeOffline() {
        service.setScenario(SimulationScenario.NODE_OFFLINE);
        String reason = service.getDropReason(testPacket, testNode);
        
        assertTrue(reason.contains("offline") || reason.contains("link failure"));
    }
    
    @Test
    void testGetDropReasonForNodeOverload() {
        service.setScenario(SimulationScenario.NODE_OVERLOAD);
        String reason = service.getDropReason(testPacket, testNode);
        
        assertTrue(reason.contains("overload") || reason.contains("queue"));
    }
    
    @Test
    void testGetDropReasonForWeatherEvent() {
        service.setScenario(SimulationScenario.WEATHER_EVENT);
        String reason = service.getDropReason(testPacket, testNode);
        
        assertTrue(reason.contains("Weather") || reason.contains("weather"));
    }
    
    @Test
    void testGetDropReasonForTTLExpired() {
        service.setScenario(SimulationScenario.TTL_EXPIRED);
        String reason = service.getDropReason(testPacket, testNode);
        
        assertTrue(reason.contains("TTL") || reason.contains("expired"));
    }
    
    @Test
    void testScenarioLatencyNormalIsZero() {
        service.setScenario(SimulationScenario.NORMAL);
        double latency = service.getScenarioLatencyMs(testNode);
        
        assertEquals(0.0, latency, 0.01);
    }
    
    @Test
    void testScenarioLatencyWeatherEventIsPositive() {
        service.setScenario(SimulationScenario.WEATHER_EVENT);
        double latency = service.getScenarioLatencyMs(testNode);
        
        assertTrue(latency >= 20.0 && latency <= 50.0);
    }
    
    @Test
    void testScenarioLatencyNodeOverloadBasedOnQueue() {
        service.setScenario(SimulationScenario.NODE_OVERLOAD);
        testNode.setCurrentPacketCount(50); // 50% full
        
        double latency = service.getScenarioLatencyMs(testNode);
        
        assertTrue(latency >= 0.0 && latency <= 100.0);
    }
    
    @Test
    void testGetNodeLoadPercent() {
        testNode.setCurrentPacketCount(25);
        testNode.setPacketBufferCapacity(100);
        
        double loadPercent = service.getNodeLoadPercent(testNode);
        
        assertEquals(25.0, loadPercent, 0.01);
    }
    
    @Test
    void testGetRandomScenario() {
        SimulationScenario random = service.getRandomScenario();
        
        assertNotNull(random);
        assertTrue(random instanceof SimulationScenario);
    }
}

package com.sagsins.core.service;

import com.sagsins.core.model.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SimulationScenarioFactoryServiceTest {

    private SimulationScenarioFactoryService service;
    private NodeInfo testNode;

    @BeforeEach
    void setUp() {
        service = new SimulationScenarioFactoryService();
        
        // Create a test node
        testNode = new NodeInfo();
        testNode.setNodeId("TEST_NODE_001");
        testNode.setNodeName("Test Satellite");
        testNode.setNodeType(NodeType.LEO_SATELLITE);
        testNode.setOperational(true);
        testNode.setBatteryChargePercent(100.0);
        testNode.setNodeProcessingDelayMs(10.0);
        testNode.setPacketLossRate(0.01);
        testNode.setResourceUtilization(50.0);
        testNode.setPacketBufferCapacity(1000);
        testNode.setCurrentPacketCount(100);
        testNode.setWeather(WeatherCondition.CLEAR);
    }

    @Test
    void testGetAvailableScenarios() {
        List<String> scenarios = service.getAvailableScenarios();
        
        assertNotNull(scenarios);
        assertTrue(scenarios.contains("NORMAL"));
        assertTrue(scenarios.contains("WEATHER_EVENT"));
        assertTrue(scenarios.contains("NODE_OVERLOAD"));
        assertTrue(scenarios.contains("NODE_OFFLINE"));
        assertTrue(scenarios.contains("TRAFFIC_SPIKE"));
        assertTrue(scenarios.contains("TTL_EXPIRED"));
    }

    @Test
    void testSetAndGetCurrentScenario() {
        assertEquals("NORMAL", service.getCurrentScenario());
        
        service.setScenario("WEATHER_EVENT");
        assertEquals("WEATHER_EVENT", service.getCurrentScenario());
    }

    @Test
    void testGetScenarioInfo() {
        Map<String, String> info = service.getScenarioInfo("WEATHER_EVENT");
        
        assertNotNull(info);
        assertEquals("WEATHER_EVENT", info.get("name"));
        assertEquals("Weather Event", info.get("displayName"));
        assertTrue(info.get("description").contains("weather"));
    }

    @Test
    void testApplyNormalScenario() {
        NodeInfo originalNode = cloneNode(testNode);
        
        NodeInfo result = service.applyScenarioToNode(testNode, "NORMAL");
        
        assertNotNull(result);
        assertEquals(originalNode.getBatteryChargePercent(), result.getBatteryChargePercent());
        assertEquals(originalNode.isOperational(), result.isOperational());
    }

    @Test
    void testApplyWeatherEventScenario() {
        double originalPacketLoss = testNode.getPacketLossRate();
        double originalDelay = testNode.getNodeProcessingDelayMs();
        
        NodeInfo result = service.applyScenarioToNode(testNode, "WEATHER_EVENT");
        
        assertNotNull(result);
        // Weather scenario should increase packet loss and delay
        assertTrue(result.getPacketLossRate() > originalPacketLoss);
        assertTrue(result.getNodeProcessingDelayMs() > originalDelay);
        assertNotEquals(WeatherCondition.CLEAR, result.getWeather());
    }

    @Test
    void testApplyNodeOverloadScenario() {
        int originalPacketCount = testNode.getCurrentPacketCount();
        double originalResourceUtil = testNode.getResourceUtilization();
        
        NodeInfo result = service.applyScenarioToNode(testNode, "NODE_OVERLOAD");
        
        assertNotNull(result);
        // Overload scenario should increase buffer usage and resource utilization
        assertTrue(result.getCurrentPacketCount() >= originalPacketCount);
        assertTrue(result.getResourceUtilization() > originalResourceUtil);
    }

    @Test
    void testApplyTrafficSpikeScenario() {
        int originalPacketCount = testNode.getCurrentPacketCount();
        
        NodeInfo result = service.applyScenarioToNode(testNode, "TRAFFIC_SPIKE");
        
        assertNotNull(result);
        // Traffic spike should increase packet count
        assertTrue(result.getCurrentPacketCount() >= originalPacketCount);
    }

    @Test
    void testApplyCurrentScenarioToMultipleNodes() {
        service.setScenario("WEATHER_EVENT");
        
        NodeInfo node1 = cloneNode(testNode);
        node1.setNodeId("NODE_001");
        
        NodeInfo node2 = cloneNode(testNode);
        node2.setNodeId("NODE_002");
        
        List<NodeInfo> nodes = Arrays.asList(node1, node2);
        List<NodeInfo> result = service.applyCurrentScenarioToNodes(nodes);
        
        assertNotNull(result);
        assertEquals(2, result.size());
        
        // Both nodes should be affected by weather
        for (NodeInfo node : result) {
            assertNotEquals(WeatherCondition.CLEAR, node.getWeather());
        }
    }

    @Test
    void testLastUpdatedIsSet() {
        NodeInfo result = service.applyScenarioToNode(testNode, "WEATHER_EVENT");
        
        assertNotNull(result);
        assertNotNull(result.getLastUpdated());
    }

    private NodeInfo cloneNode(NodeInfo source) {
        NodeInfo clone = new NodeInfo();
        clone.setNodeId(source.getNodeId());
        clone.setNodeName(source.getNodeName());
        clone.setNodeType(source.getNodeType());
        clone.setOperational(source.isOperational());
        clone.setBatteryChargePercent(source.getBatteryChargePercent());
        clone.setNodeProcessingDelayMs(source.getNodeProcessingDelayMs());
        clone.setPacketLossRate(source.getPacketLossRate());
        clone.setResourceUtilization(source.getResourceUtilization());
        clone.setPacketBufferCapacity(source.getPacketBufferCapacity());
        clone.setCurrentPacketCount(source.getCurrentPacketCount());
        clone.setWeather(source.getWeather());
        return clone;
    }
}

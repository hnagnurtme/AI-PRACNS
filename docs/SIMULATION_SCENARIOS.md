# Simulation Scenarios Guide

## Overview

The SAGSINS platform now supports diverse simulation scenarios to test network behavior under different conditions. These scenarios help validate the performance of RL-based routing versus Dijkstra routing under stress conditions.

## Available Scenarios

### 1. Normal (Default)
- **Description**: Standard network operation with no special conditions
- **Use Case**: Baseline performance testing
- **Effects**: None - normal packet routing and processing

### 2. Weather Event
- **Description**: Bad weather affecting transmission quality
- **Use Case**: Testing network resilience during atmospheric interference
- **Effects**:
  - Randomly sets nodes to STORM, RAIN, or SEVERE_STORM weather conditions
  - Adds 20-50ms additional latency
  - Increases packet loss rate by 15% during severe weather
  - Simulates signal attenuation and increased noise

### 3. Node Overload
- **Description**: Nodes experiencing high load and queue congestion
- **Use Case**: Testing behavior during traffic surges
- **Effects**:
  - Sets node queues to 70-95% full capacity
  - Adds latency proportional to queue size (up to 100ms)
  - 30% chance of packet drop when queue exceeds 90%
  - Simulates buffer overflow conditions

### 4. Node Offline
- **Description**: Nodes temporarily offline or unreachable
- **Use Case**: Testing fault tolerance and route recovery
- **Effects**:
  - Randomly marks 20% of nodes as offline
  - 50% chance of packet drop at offline nodes
  - Simulates link failures and node downtime
  - Tests routing algorithm adaptability

### 5. Traffic Spike
- **Description**: Sudden burst of traffic causing congestion
- **Use Case**: Testing handling of concurrent packet floods
- **Effects**:
  - Sets node queues to 50-90% full capacity
  - Adds 10-30ms additional latency
  - Simulates flash crowd scenarios
  - Tests load balancing capabilities

### 6. TTL Expired
- **Description**: Packet dropped due to time-to-live expiration
- **Use Case**: Testing packet lifetime management
- **Effects**:
  - Drops packets when TTL reaches 0
  - Simulates routing loops and excessive hops
  - Tests packet freshness requirements

## Usage

### Via Frontend UI

1. Navigate to the Monitor/Dashboard page
2. Locate the **Simulation Scenario** dropdown at the top
3. Select your desired scenario from the dropdown
4. The scenario will be applied immediately to new packet simulations
5. Click "Reset to Normal" to return to default behavior

### Via REST API

#### List Available Scenarios
```bash
GET http://localhost:8080/api/simulation/scenarios
```

Response:
```json
["NORMAL", "WEATHER_EVENT", "NODE_OVERLOAD", "NODE_OFFLINE", "TRAFFIC_SPIKE", "TTL_EXPIRED"]
```

#### Get Current Scenario
```bash
GET http://localhost:8080/api/simulation/scenario/current
```

Response:
```json
{
  "scenario": "WEATHER_EVENT",
  "displayName": "Weather Event",
  "description": "Bad weather affecting transmission quality"
}
```

#### Set Scenario
```bash
POST http://localhost:8080/api/simulation/scenario/WEATHER_EVENT
Content-Type: application/json
```

Response:
```json
{
  "success": true,
  "previousScenario": "NORMAL",
  "currentScenario": "WEATHER_EVENT",
  "message": "Scenario changed to: Weather Event"
}
```

#### Reset to Normal
```bash
POST http://localhost:8080/api/simulation/scenario/reset
Content-Type: application/json
```

Response:
```json
{
  "success": true,
  "previousScenario": "WEATHER_EVENT",
  "currentScenario": "NORMAL",
  "message": "Scenario reset to NORMAL"
}
```

## Observing Scenario Effects

### In Packet Hop Records

Each hop in a packet's journey now includes scenario information:

```json
{
  "fromNodeId": "LEO-SAT-001",
  "toNodeId": "GEO-SAT-005",
  "latencyMs": 85.5,
  "scenarioType": "WEATHER_EVENT",
  "nodeLoadPercent": 75.2,
  "dropReasonDetails": "Weather event - poor transmission conditions (STORM)"
}
```

### In UI Visualizations

- **Hop Tooltips**: Hover over route visualization to see scenario details
- **Scenario Badge**: Yellow warning badge appears for non-normal scenarios
- **Node Load**: Visual indication of buffer utilization percentage
- **Drop Reasons**: Detailed explanation when packets are dropped

## Performance Comparison

The PacketComparison results include scenario context:

```
═══════════════════════════════════════════════════
🏁 PACKET COMPARISON COMPLETE: user1_user2_1699876543
───────────────────────────────────────────────────
📍 Route: user1 → user2

📊 DIJKSTRA:
   • Route Latency:  125.50 ms (from AnalysisData)
   • Hops:           5
   • Dropped:        NO

🤖 REINFORCEMENT LEARNING:
   • Route Latency:  98.25 ms (from AnalysisData)
   • Hops:           4
   • Dropped:        NO

🏆 Winner: RL (21.72% faster)
🎬 Scenario: WEATHER_EVENT
═══════════════════════════════════════════════════
```

## Testing Recommendations

### 1. Baseline Testing
- Run with **NORMAL** scenario first
- Establish baseline metrics (latency, hops, success rate)
- Document expected behavior

### 2. Stress Testing
- Apply **NODE_OVERLOAD** or **TRAFFIC_SPIKE**
- Send multiple concurrent packets
- Measure degradation gracefully

### 3. Resilience Testing
- Apply **NODE_OFFLINE** scenario
- Verify route recovery and alternate path selection
- Test both RL and Dijkstra recovery times

### 4. Environmental Testing
- Apply **WEATHER_EVENT** scenario
- Compare RL vs Dijkstra adaptation to conditions
- Measure impact on latency and packet loss

### 5. Lifetime Testing
- Apply **TTL_EXPIRED** scenario
- Verify packet freshness enforcement
- Test scenarios with varying TTL values

## Integration with Existing Features

### Batch Simulations
- Scenarios apply to entire batches
- All packet pairs experience same scenario conditions
- Useful for consistent comparative analysis

### Real-time Monitoring
- Scenario changes reflect immediately in WebSocket updates
- Dashboard updates show current scenario status
- Historical data maintains scenario context

### Analysis and Reporting
- Scenario metadata included in all packet records
- MongoDB collections store scenario information
- Query and filter by scenario type for analysis

## Best Practices

1. **Reset Between Tests**: Use "Reset to Normal" between different test runs
2. **Document Scenarios**: Record which scenario was active for each test
3. **Compare Apples to Apples**: Run same scenario for RL and Dijkstra comparison
4. **Monitor Node States**: Check node buffer and operational status during scenarios
5. **Analyze Drop Reasons**: Use detailed drop reasons to understand failures

## Troubleshooting

### Scenario Not Applying
- Check that backend services (SAGSINs-core) are running
- Verify API endpoint is accessible at http://localhost:8080
- Check browser console for any error messages

### UI Not Updating
- Ensure WebSocket connection is active
- Check VITE_API_URL environment variable
- Refresh the page to re-establish connections

### Inconsistent Behavior
- Scenario effects are probabilistic (due to randomness)
- Run multiple tests to establish patterns
- Use larger sample sizes for statistical significance

## Future Enhancements

Potential additions for future versions:
- Custom scenario parameters (e.g., adjustable packet loss rates)
- Scheduled scenario transitions
- Scenario recording and playback
- Geographic-based scenarios
- Multi-scenario combinations

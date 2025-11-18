# Batch Monitor Visualization Fixes

## Date: 2025-11-19

## Problems Fixed

### 1. ❌ Route Visualization Error
**Issue**: Frontend PacketRouteGraph component couldn't render routes because `hopRecords` were missing position data (`fromNodePosition` and `toNodePosition`).

**Root Cause**: The Python test script (`test_batch_packet_service_beautiful.py`) was creating `HopRecord` objects without position coordinates.

**Solution**:
- ✅ Added `from_coords` and `to_coords` parameters to `create_hop_record()` function
- ✅ Created `Position` objects for each hop with latitude, longitude, and altitude
- ✅ Stations use their defined coordinates from `CITIES` dict
- ✅ Satellite nodes get random coordinates interpolated between source and destination

**Files Modified**:
- `src/rl-router/test_batch_packet_service_beautiful.py` (lines 125-180, 195-240)

### 2. ❌ AttributeError: 'str' object has no attribute 'value'
**Issue**: `RoutingDecisionInfo.algorithm` was being created as a string instead of using the `RoutingAlgorithm` enum.

**Root Cause**: The test script passed strings like `"ReinforcementLearning"` and `"Dijkstra"` to the algorithm parameter, but the `to_dict()` method expected an enum with a `.value` attribute.

**Solution**:
- ✅ Import `RoutingAlgorithm` enum in test script
- ✅ Use `RoutingAlgorithm.REINFORCEMENT_LEARNING` and `RoutingAlgorithm.DIJKSTRA` enums
- ✅ Made `_hop_record_to_dict()` method more robust with type checking

**Files Modified**:
- `src/rl-router/test_batch_packet_service_beautiful.py` (line 17, 176)
- `src/rl-router/model/Packet.py` (line 165)

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Test Script                          │
│           (test_batch_packet_service_beautiful.py)             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Create BatchPacket with:                                     │
│    - dijkstraPacket with HopRecords (including positions)       │
│    - rlPacket with HopRecords (including positions)             │
│ 2. Save to MongoDB via BatchPacketService                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MongoDB Collections                         │
│           - two_packets                                         │
│           - batch_packets                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ (Change Stream)
┌─────────────────────────────────────────────────────────────────┐
│              Java Backend Service                               │
│          (PacketChangeStreamService)                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Detects changes in batch_packets collection                  │
│ 2. Waits 3 seconds from last update                             │
│ 3. Sends BatchPacket to WebSocket: /topic/batchpacket           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ (WebSocket)
┌─────────────────────────────────────────────────────────────────┐
│              Frontend React App                                 │
│         (BatchMonitor.tsx + useBatchWebSocket)                  │
├─────────────────────────────────────────────────────────────────┤
│ 1. Receives BatchPacket via WebSocket                           │
│ 2. Calculates congestion map from hopRecords                    │
│ 3. Renders:                                                     │
│    - BatchStatistics (overview)                                 │
│    - NetworkTopologyView (node congestion cards)                │
│    - PacketFlowDetail (detailed packet info)                    │
│    - AlgorithmComparisonChart (performance comparison)          │
│    - PacketRouteGraph (visual route with positions) ✅ NOW WORKS│
└─────────────────────────────────────────────────────────────────┘
```

## Required Data Structure

For route visualization to work, each `HopRecord` must have:

```typescript
{
  fromNodeId: string,
  toNodeId: string,
  latencyMs: number,
  timestampMs: number,
  distanceKm: number,
  packetLossRate: number,
  fromNodePosition: {          // ✅ REQUIRED for visualization
    latitude: number,
    longitude: number,
    altitude: number
  },
  toNodePosition: {            // ✅ REQUIRED for visualization
    latitude: number,
    longitude: number,
    altitude: number
  },
  fromNodeBufferState: {
    queueSize: number,
    bandwidthUtilization: number  // 0.0 - 1.0
  },
  routingDecisionInfo: {
    algorithm: "Dijkstra" | "ReinforcementLearning",  // ✅ MUST be enum value
    metric: string,
    reward: number (optional)
  }
}
```

## How to Test

### 1. Start the test script (choose Scenario 1):
```bash
cd /Users/anhnon/PBL4/src/rl-router
python test_batch_packet_service_beautiful.py
# Select option 1 (Perfect Comparison)
```

### 2. Check Java backend logs:
```bash
# Look for these messages:
# 🔔 [CHANGE EVENT] Received change event for batch_packets collection!
# 📤 [SENT] BatchPacket to /topic/batchpacket
```

### 3. Open frontend:
```
http://localhost:3000/batch-monitor
```

### Expected Results:
✅ Connection status shows "CONNECTED"
✅ Batch statistics display correctly
✅ Network topology shows nodes with congestion levels
✅ Route visualization displays with:
   - Nodes positioned based on lat/lon coordinates
   - Edges showing packet hops with latency labels
   - Node sizes reflecting bandwidth utilization
   - Purple nodes for queued packets
   - Red nodes for dropped packets (if any)
✅ Algorithm comparison charts show performance metrics

## Common Issues & Solutions

### Issue: "No route data available"
**Solution**: Check that `hopRecords` contain `fromNodePosition` and `toNodePosition`. The updated test script now includes these automatically.

### Issue: Frontend doesn't receive data
**Solutions**:
1. Check MongoDB connection in Java backend
2. Verify WebSocket endpoint: `ws://localhost:8080/ws`
3. Check browser console for WebSocket connection errors
4. Ensure CORS is properly configured

### Issue: Routes display but positions are wrong
**Solution**: Verify latitude/longitude values are realistic:
- Hanoi: (21.0285, 105.8542)
- Bangkok: (13.7563, 100.5018)
- Singapore: (1.3521, 103.8198)
- Tokyo: (35.6762, 139.6503)
- Seoul: (37.5665, 126.9780)

### Issue: "Simulator scenario ko hoat dong" (Simulator scenario not working)
**Possible causes**:
1. Check if Java backend simulation endpoints are running
2. Verify frontend routing to batch monitor page
3. Check if scenario selection component exists

**Next steps**: Need to investigate scenario selection mechanism (see below)

## Remaining Work

### Simulator Scenario Investigation
Need to check:
1. ✅ Backend scenario API endpoints
2. ✅ Frontend scenario selection UI
3. ✅ Integration between scenario selection and batch monitor

## Verification Checklist

- [x] Position data included in HopRecords
- [x] RoutingAlgorithm enum used correctly
- [x] to_dict() method handles enum values
- [x] Backend sends data via WebSocket
- [x] Frontend receives and parses data
- [x] Route visualization renders correctly
- [ ] Simulator scenario functionality works
- [ ] End-to-end test with real network simulation

## Performance Notes

- Backend batches updates (3 second delay from last change)
- Frontend uses useMemo for expensive calculations
- Congestion map recalculates only when batch changes
- Route normalization cached per data update

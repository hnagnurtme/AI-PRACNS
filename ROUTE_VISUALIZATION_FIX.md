# Route Visualization Fix

## Issue
Route Visualization & Hop Performance không hiển thị trên batch monitor page.

## Root Cause
Component `PacketRouteGraph` không được import và sử dụng trong `BatchMonitor.tsx`.

## Solution Applied

### 1. Import Component
Added import statement in [BatchMonitor.tsx:9](src/sagsins-frontend/src/pages/BatchMonitor.tsx#L9):
```typescript
import { PacketRouteGraph } from '../components/chart/PacketRouteGraph';
```

### 2. Add State for Packet Pair Selection
Added state to track which packet pair to visualize (line 24):
```typescript
const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0);
```

### 3. Add Route Visualization Section
Added complete route visualization section with selector (lines 77-104):

```tsx
{/* 4. Route Visualization & Hop Performance */}
{latestBatch && latestBatch.packets && latestBatch.packets.length > 0 && (
    <div className="space-y-4">
        {/* Packet Pair Selector - Only show if multiple pairs */}
        {latestBatch.packets.length > 1 && (
            <div className="bg-white rounded-lg border border-gray-300 p-4">
                <label>Select Packet Pair to Visualize Route</label>
                <select onChange={(e) => setSelectedPairIndex(Number(e.target.value))}>
                    {latestBatch.packets.map((pair, idx) => (
                        <option key={idx} value={idx}>
                            Pair #{idx + 1} - {source} → {dest}
                        </option>
                    ))}
                </select>
            </div>
        )}

        {/* Route Visualization for Selected Pair */}
        <PacketRouteGraph data={latestBatch.packets[selectedPairIndex]} />
    </div>
)}
```

## Features Added

### 1. Packet Pair Selector
- Shows dropdown when batch has multiple packet pairs
- Displays source → destination for each pair
- User can select which pair to visualize
- Defaults to Pair #1

### 2. Route Visualization
- Shows PacketRouteGraph for selected packet pair
- Displays both Dijkstra and RL routes
- Geographic node positioning
- Hop latency and distance labels
- Node buffer state indicators
- Toggle buttons for each algorithm

## How It Works

1. **Data Flow**:
   ```
   latestBatch.packets[selectedPairIndex]
        ↓
   ComparisonData { dijkstraPacket, rlPacket }
        ↓
   PacketRouteGraph component
        ↓
   Visual route map with nodes and hops
   ```

2. **Packet Pair Structure**:
   ```typescript
   {
     dijkstraPacket: {
       packetId: "PKT_DIJKSTRA_000",
       stationSource: "STATION_HANOI",
       stationDest: "STATION_BANGKOK",
       hopRecords: [
         {
           fromNodeId: "STATION_HANOI",
           toNodeId: "SAT_LEO_001",
           fromNodePosition: { lat: 21.0285, lon: 105.8542, alt: 10000 },
           toNodePosition: { lat: 18.5, lon: 103.2, alt: 850 },
           ...
         },
         ...
       ]
     },
     rlPacket: { ... }
   }
   ```

3. **Visualization Features**:
   - Node circles sized by bandwidth utilization
   - Edge colors based on hop latency (green=low, red=high)
   - Labels showing latency (ms) and distance (km)
   - Hover effects highlighting connections
   - Drop indicators (red nodes with pulse)

## Testing

### 1. Start System
```bash
# Backend
cd /Users/anhnon/PBL4/src/SAGSINs
./mvnw spring-boot:run

# Frontend
cd /Users/anhnon/PBL4/src/sagsins-frontend
npm run dev
```

### 2. Run Test Script
```bash
cd /Users/anhnon/PBL4/src/rl-router
python test_batch_packet_service_beautiful.py
# Select: 1 (Perfect Comparison - 5 pairs)
```

### 3. Open Batch Monitor
```
http://localhost:3000/batch-monitor
```

### 4. Expected Results ✅

1. **Connection shows**: "CONNECTED"
2. **Scenario Selector** appears at top
3. **Batch Statistics** displays
4. **Network Topology** shows nodes
5. **Algorithm Comparison** shows charts
6. **NEW: Packet Pair Selector** appears with dropdown showing:
   ```
   Pair #1 - STATION_HANOI → STATION_BANGKOK
   Pair #2 - STATION_HANOI → STATION_BANGKOK
   Pair #3 - STATION_HANOI → STATION_BANGKOK
   Pair #4 - STATION_HANOI → STATION_BANGKOK
   Pair #5 - STATION_HANOI → STATION_BANGKOK
   ```
7. **Route Visualization** displays below with:
   - Title: "🗺️ Route Visualization & Hop Performance"
   - Toggle buttons: "Dijkstra" (blue) and "RL" (orange)
   - Map showing nodes positioned geographically
   - Edges connecting nodes with arrows
   - Labels on edges showing latency and distance
   - Bottom section showing overall stats and drop reasons

### 5. Interaction Testing

**Test Pair Selection**:
1. Change dropdown to "Pair #2"
2. Route visualization updates
3. New source/destination route displays
4. Metrics update accordingly

**Test Route Toggles**:
1. Click "Dijkstra" button to hide Dijkstra route
2. Click "RL" button to hide RL route
3. Both can be toggled independently

**Test Node Hover**:
1. Hover over any node circle
2. Blue highlight appears around node
3. Connected edges become more prominent

## Troubleshooting

### Issue: Route visualization still not showing

**Check 1: Browser Console**
```
F12 → Console tab
Look for errors like:
- "Cannot read property 'hopRecords' of undefined"
- "Position is null"
```

**Check 2: Data Structure**
```
F12 → Console tab
Type: latestBatch.packets[0]
Verify structure has:
- dijkstraPacket
- rlPacket
- Each with hopRecords[]
- Each hopRecord has fromNodePosition and toNodePosition
```

**Check 3: React DevTools**
```
Install React DevTools extension
Check BatchDashboard component state:
- latestBatch should not be null
- packets array should have items
- selectedPairIndex should be 0
```

### Issue: "No route data available"

**This means the selected packet pair is missing data.**

**Fix**:
1. Re-run test script with latest code
2. Verify positions are included in HopRecords
3. Check MongoDB data:
   ```bash
   mongo rl_router
   db.batch_packets.findOne({}, {packets: {$slice: 1}})
   # Verify hopRecords have fromNodePosition and toNodePosition
   ```

### Issue: Selector shows but no pairs listed

**This means latestBatch.packets is empty.**

**Fix**:
1. Check backend logs for successful BatchPacket send
2. Verify Change Stream is working
3. Check that test script creates packets correctly
4. Wait 3-5 seconds for batching delay

## Files Modified

- [BatchMonitor.tsx](src/sagsins-frontend/src/pages/BatchMonitor.tsx)
  - Line 9: Import PacketRouteGraph
  - Line 24: Add selectedPairIndex state
  - Lines 77-104: Add route visualization section

## Summary

✅ **Route Visualization is now fully integrated into Batch Monitor**

Features:
- Packet pair selector (when multiple pairs exist)
- Geographic route visualization
- Toggle between Dijkstra and RL routes
- Detailed hop performance metrics
- Interactive hover effects
- Drop indicators

The component now appears below the Algorithm Comparison Chart and provides complete visual analysis of packet routes through the network.

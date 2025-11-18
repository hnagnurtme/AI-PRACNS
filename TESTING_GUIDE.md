# Testing Guide - Beautiful Sample Data

## 🎨 Overview

This guide shows how to test the entire system with beautiful, realistic sample data.

---

## 🚀 Quick Start

### 1. Start Backend

```bash
cd src/SAGSINs
./mvnw spring-boot:run
```

**Wait for:**
```
✅ MongoDB Change Stream listeners started successfully
✅ MessageListenerContainer is RUNNING
🎯 Ready to receive change events from MongoDB
```

### 2. Start Frontend

```bash
cd src/sagsins-frontend
npm start
```

Open browser: `http://localhost:3000`

### 3. Run Beautiful Test Script

```bash
cd /Users/anhnon/PBL4
python3 test_batch_packet_service_beautiful.py
```

---

## 📊 Test Scenarios

### Scenario 1: Perfect Comparison ✅

**Description:** Both algorithms work perfectly, easy comparison

**Route:** HANOI → BANGKOK
**Packets:** 5 pairs (10 total)
**Service:** VIDEO_STREAMING
**Expected:**
- ✅ All packets delivered
- Low latency for both
- Clear comparison metrics

**Use Case:** Verify basic functionality

---

### Scenario 2: RL Advantage 🎯

**Description:** RL outperforms Dijkstra

**Route:** SINGAPORE → TOKYO
**Packets:** 5 pairs (10 total)
**Service:** AUDIO_CALL
**Expected:**
- ❌ Dijkstra: Some drops (every 3rd packet)
- ✅ RL: All delivered
- RL shows lower average latency

**Use Case:** Demonstrate RL benefits

---

### Scenario 3: Mixed Services 🎭

**Description:** Different QoS requirements

**Route:** SEOUL → HANOI
**Packets:** 4 pairs (8 total)
**Services:**
- VIDEO_STREAMING (max 150ms latency)
- AUDIO_CALL (max 100ms latency)
- IMAGE_TRANSFER (max 200ms latency)
- TEXT_MESSAGE (max 500ms latency)

**Expected:**
- Different latency tolerances
- Algorithm adaptation to QoS
- Service-specific metrics

**Use Case:** Test QoS handling

---

### Scenario 4: High Load 🔥

**Description:** Stress test with congestion

**Route:** BANGKOK → SINGAPORE
**Packets:** 10 pairs (20 total)
**Service:** VIDEO_STREAMING
**Expected:**
- High network congestion
- Dijkstra: ~20% drop rate
- RL: ~10% drop rate (better load balancing)
- Varied latencies

**Use Case:** Congestion visualization

---

## 🎯 What to Check

### Backend Logs:

```bash
# Should see these logs for each packet:

🔔 [CHANGE EVENT] Received change event for two_packets collection!
📝 Operation Type: REPLACE
🔄 [REPLACE] TwoPacket received - pairId=USER_HANOI_USER_BANGKOK, dijkstra=✓, rl=✓, complete=YES
⏰ Scheduled TwoPacket send in 3000ms - pairId=USER_HANOI_USER_BANGKOK

# After 3 seconds:
📤 [SENT] TwoPacket to /topic/packets - pairId=USER_HANOI_USER_BANGKOK, dijkstra=PKT_DIJKSTRA_000, rl=PKT_RL_000

# After 10 seconds:
🗑️ [DELETED] TwoPacket - pairId=USER_HANOI_USER_BANGKOK (after 10000ms)
```

### Frontend Monitor Page:

Navigate to: `http://localhost:3000/monitor`

**Should Display:**

1. **PacketRouteGraph:**
   - Two side-by-side route visualizations
   - Left: Dijkstra route (blue)
   - Right: RL route (purple)
   - Nodes and connections

2. **CombinedHopMetricsChart:**
   - Latency comparison
   - Distance comparison
   - Buffer state visualization

3. **Real-time updates:**
   - New data every 3-5 seconds
   - Smooth transitions
   - Correct algorithm labels

### Frontend Batch Monitor Page:

Navigate to: `http://localhost:3000/batch-monitor`

**Should Display:**

1. **BatchChart:**
   - List of received batches
   - Packet pair count
   - Algorithm distribution

2. **NodeCongestionCard:**
   - Click on nodes to see details
   - Packets routed through each node
   - Queue sizes and bandwidth utilization
   - Algorithm breakdown

---

## 🐛 Troubleshooting

### No data appears in frontend

**Check:**
1. ✅ Backend is running and logs show "✅ MessageListenerContainer is RUNNING"
2. ✅ Test script completed successfully
3. ✅ Wait at least 3-5 seconds for WebSocket updates
4. ✅ Browser console shows "📩 Packet message received"

**Solution:**
- Refresh page
- Check WebSocket connection in Network tab
- Verify backend logs show "📤 [SENT]"

### Backend shows incomplete packets

**Log:**
```
⏸️ TwoPacket incomplete - pairId=..., waiting for both packets
```

**Cause:** Only one packet (Dijkstra OR RL) received, not both

**Solution:**
- Ensure test script sends BOTH packets
- Check MongoDB has both packets:
  ```javascript
  db.two_packets.find({pairId: "USER_HANOI_USER_BANGKOK"})
  ```
- Look for `dijkstraPacket` AND `rlPacket` fields

### Field name errors in console

**Error:**
```javascript
packet.useRL is undefined
```

**Cause:** Old code, not updated

**Solution:**
- Ensure all fixes from [FRONTEND_BACKEND_SYNC_FIX.md](FRONTEND_BACKEND_SYNC_FIX.md) applied
- Clear browser cache: Ctrl+Shift+R or Cmd+Shift+R
- Rebuild frontend: `npm run build`

---

## 📈 Expected Metrics

### Scenario 1 (Perfect Comparison):

```
Dijkstra:
- Success Rate: 100%
- Avg Latency: 60-100ms
- Packets: 5/5

RL:
- Success Rate: 100%
- Avg Latency: 50-90ms (slightly better)
- Packets: 5/5
```

### Scenario 2 (RL Advantage):

```
Dijkstra:
- Success Rate: ~60-70%
- Avg Latency: 80-120ms
- Dropped: 1-2 packets

RL:
- Success Rate: 100%
- Avg Latency: 60-90ms
- Dropped: 0 packets
```

### Scenario 3 (Mixed Services):

```
Each service shows different latency characteristics:
- VIDEO_STREAMING: Lower latency
- AUDIO_CALL: Lowest latency (high priority)
- IMAGE_TRANSFER: Moderate latency
- TEXT_MESSAGE: Higher latency tolerance
```

### Scenario 4 (High Load):

```
Dijkstra:
- Success Rate: ~80%
- Avg Latency: 100-150ms
- Shows congestion impact

RL:
- Success Rate: ~90%
- Avg Latency: 80-120ms
- Better load balancing
```

---

## 🎨 Data Features

The beautiful test data includes:

### ✅ Realistic Routing:
- Actual satellite node IDs (SAT_LEO_001-008)
- Asian cities network (Hanoi, Bangkok, Singapore, Tokyo, Seoul)
- Variable hop counts (2-5 hops)

### ✅ Network Congestion:
- Random congestion points in routes
- Higher latency when congested
- Queue sizes: 0-100
- Bandwidth utilization: 10-95%

### ✅ QoS Handling:
- 4 service types with different requirements
- Priority levels (1-5)
- Latency thresholds
- Loss rate limits

### ✅ Algorithm Differences:
- RL: More hops, better load balancing
- Dijkstra: Fewer hops, shortest path
- Different success rates under load

### ✅ Visualization-Friendly:
- Clear route differences
- Observable metric variations
- Congestion hotspots
- Algorithm performance comparison

---

## 🔗 Related Documents

- [FRONTEND_BACKEND_SYNC_FIX.md](FRONTEND_BACKEND_SYNC_FIX.md) - Field name fixes
- [FRONTEND_MONITOR_TROUBLESHOOTING.md](FRONTEND_MONITOR_TROUBLESHOOTING.md) - Debug guide
- [CHANGESTREAM_FIX_SUMMARY.md](CHANGESTREAM_FIX_SUMMARY.md) - Change Stream setup

---

## 💡 Tips

1. **Run scenarios in order** (1 → 2 → 3 → 4) to see progression

2. **Wait between scenarios** to avoid overwhelming the system

3. **Clean MongoDB between runs:**
   ```javascript
   db.two_packets.deleteMany({})
   db.batch_packets.deleteMany({})
   ```

4. **Monitor backend logs** to understand data flow

5. **Use browser DevTools** to inspect WebSocket messages

6. **Take screenshots** of good visualizations for documentation

---

## 🎉 Success Criteria

✅ All scenarios run without errors
✅ Backend logs show proper event flow
✅ Frontend displays all packets
✅ Metrics are realistic and varied
✅ Algorithm differences are visible
✅ Visualizations are clear and beautiful

---

Enjoy testing with beautiful data! 🚀

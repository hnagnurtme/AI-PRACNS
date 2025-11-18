# Frontend Monitor Troubleshooting Guide

## 🔍 Vấn đề

Server chưa gửi TwoPacket, frontend chưa hiển thị monitor.

---

## 📊 Data Flow Analysis

### Backend Flow:

```
Python BatchPacketService
  ↓ save_packet()
  ↓ MongoDB two_packets collection
  ↓ Change Stream event (INSERT/REPLACE)
  ↓ PacketChangeStreamService.handleTwoPacketChange()
  ↓ Check: hasBothPackets? (dijkstra AND rl)
  ↓ Wait 3 seconds (SEND_DELAY_MS)
  ↓ messagingTemplate.convertAndSend("/topic/packets", twoPacket)
  ↓ WebSocket → Frontend
```

### Frontend Flow:

```
usePacketWebSocket hook
  ↓ Subscribe to "/topic/packets"
  ↓ Receive TwoPacket (ComparisonData)
  ↓ setMessages([...prev, body])
  ↓ Monitor.tsx renders latest packet
  ↓ PacketRouteGraph + CombinedHopMetricsChart
```

---

## ✅ Checklist để debug

### 1. Kiểm tra Backend có chạy không

```bash
# Check Java application logs
cd src/SAGSINs
./mvnw spring-boot:run

# Should see:
# ✅ MongoDB Change Stream listeners started successfully
# ✅ MessageListenerContainer is RUNNING
# 🎯 Ready to receive change events from MongoDB
```

### 2. Kiểm tra MongoDB Change Stream hoạt động

**Check Java logs cho:**
```
🔔 [CHANGE EVENT] Received change event for two_packets collection!
📝 Operation Type: INSERT
⏰ Scheduled TwoPacket send in 3000ms - pairId=...
```

**Nếu KHÔNG thấy logs này:**
- Change Stream chưa nhận được event
- Có thể Python chưa lưu packet
- Hoặc MongoDB connection issue

**Run test script:**
```bash
cd /Users/anhnon/PBL4
python3 test_change_stream.py
```

### 3. Kiểm tra có gửi WebSocket message không

**Check Java logs cho:**
```
📤 [SENT] TwoPacket to /topic/packets - pairId=..., dijkstra=..., rl=...
```

**Nếu KHÔNG thấy logs này:**

**Possible reasons:**

a. **Chưa đủ 2 packets:**
```
⏸️ TwoPacket incomplete - pairId=..., waiting for both packets
```
→ Cần cả Dijkstra packet VÀ RL packet

b. **Bị cancel vì có update mới:**
```
⏹️ Cancelled previous TwoPacket send task
```
→ Có packet mới arrive trong 3 seconds window

c. **Validation failed:**
```
⚠️ TwoPacket incomplete at send time - pairId=..., skipping send
```
→ Double-check failed

### 4. Kiểm tra Frontend WebSocket connection

**Browser Console logs:**
```javascript
// Should see:
✅ Connected to Packet WebSocket
📩 Packet message received
```

**Nếu KHÔNG thấy:**

a. **Check WebSocket URL:**
```typescript
// Monitor.tsx:9
const packets = usePacketWebSocket(import.meta.env.VITE_WS_URL);
```

**Check `.env` file:**
```bash
cat src/sagsins-frontend/.env
```

Should have:
```
VITE_WS_URL=http://localhost:8080/ws
```

b. **Test connection manually:**
```javascript
// Open browser console on http://localhost:3000
const socket = new SockJS('http://localhost:8080/ws');
const client = new Stomp.Client({
  webSocketFactory: () => socket,
  onConnect: () => console.log('Connected!'),
});
client.activate();
```

### 5. Kiểm tra data format

**In browser console:**
```javascript
// Should see ComparisonData structure:
{
  dijkstraPacket: {
    packetId: "...",
    isUseRL: false,  // ✅ Must be false
    // ...
  },
  rlPacket: {
    packetId: "...",
    isUseRL: true,   // ✅ Must be true
    // ...
  }
}
```

**Nếu thấy field name sai:**
- Check [FRONTEND_BACKEND_SYNC_FIX.md](FRONTEND_BACKEND_SYNC_FIX.md)
- Ensure đã fix `useRL` → `isUseRL`

---

## 🛠️ Common Issues & Solutions

### Issue 1: "Waiting for packet data..." không biến mất

**Cause:** Frontend không nhận được WebSocket message

**Solutions:**

1. **Check backend logs** - có gửi message không?
2. **Check browser console** - có connect được không?
3. **Check URL** - đúng `http://localhost:8080/ws`?
4. **Check CORS** - backend có allow frontend origin không?

### Issue 2: Backend gửi message nhưng frontend không nhận

**Cause:** WebSocket connection issue

**Solutions:**

1. **Restart both services:**
   ```bash
   # Terminal 1: Backend
   cd src/SAGSINs
   ./mvnw spring-boot:run

   # Terminal 2: Frontend
   cd src/sagsins-frontend
   npm start
   ```

2. **Check firewall/antivirus** blocking WebSocket

3. **Try different port:**
   ```bash
   # Backend application.properties
   server.port=8081

   # Frontend .env
   VITE_WS_URL=http://localhost:8081/ws
   ```

### Issue 3: Backend không gửi TwoPacket (chỉ nhận Change Event)

**Cause:** Chưa đủ 2 packets (dijkstra AND rl)

**Solutions:**

1. **Check Python sends BOTH packets:**
   ```python
   # Must send both:
   # 1. Packet with use_rl=False (Dijkstra)
   # 2. Packet with use_rl=True (RL)
   # Same source_user_id and destination_user_id
   ```

2. **Check Java logs:**
   ```
   🔄 [INSERT] TwoPacket received - pairId=..., dijkstra=✓, rl=✗, complete=NO
   ⏸️ TwoPacket incomplete - waiting for both packets
   ```
   → Need to send the missing packet

3. **Manual test:**
   ```bash
   python3 test_change_stream.py
   ```
   This sends both packets automatically

### Issue 4: Monitor hiển thị rồi nhưng data sai

**Cause:** Field name mismatch

**Check:**
```javascript
// Browser console
console.log(packets[0]);

// Should have:
packets[0].dijkstraPacket.isUseRL === false  // ✅
packets[0].rlPacket.isUseRL === true         // ✅

// NOT:
packets[0].dijkstraPacket.useRL === undefined  // ❌
```

**Solution:** See [FRONTEND_BACKEND_SYNC_FIX.md](FRONTEND_BACKEND_SYNC_FIX.md)

---

## 🧪 Step-by-step Testing

### Test 1: Backend Change Stream

```bash
# Terminal 1: Start backend
cd src/SAGSINs
./mvnw spring-boot:run

# Terminal 2: Run test
cd /Users/anhnon/PBL4
python3 test_change_stream.py
```

**Expected backend logs:**
```
🔔 [CHANGE EVENT] Received change event for two_packets collection!
📝 Operation Type: INSERT
⏰ Scheduled TwoPacket send in 3000ms - pairId=USER_HANOI_USER_BANGKOK
(Wait 3 seconds)
🔔 [CHANGE EVENT] Received change event for two_packets collection!
📝 Operation Type: REPLACE
⏰ Scheduled TwoPacket send in 3000ms - pairId=USER_HANOI_USER_BANGKOK
(Wait 3 seconds)
📤 [SENT] TwoPacket to /topic/packets - pairId=USER_HANOI_USER_BANGKOK
```

### Test 2: Frontend WebSocket

```bash
# Terminal 3: Start frontend
cd src/sagsins-frontend
npm start
```

Open browser: `http://localhost:3000`

Navigate to Monitor page

**Expected browser console:**
```
✅ Connected to Packet WebSocket
📩 Packet message received
```

**Expected UI:**
- PacketRouteGraph showing 2 routes (Dijkstra vs RL)
- CombinedHopMetricsChart showing metrics

### Test 3: End-to-end with real Python service

```bash
# Terminal 1: Backend
cd src/SAGSINs
./mvnw spring-boot:run

# Terminal 2: Python rl-router
cd src/rl-router
python service/TCPReciever.py

# Terminal 3: Frontend
cd src/sagsins-frontend
npm start

# Terminal 4: Send test packet
# (Use your packet sending method)
```

---

## 📝 Debug Logs to Enable

### Backend (application.properties):

```properties
logging.level.com.sagsins.core.service.PacketChangeStreamService=DEBUG
logging.level.org.springframework.web.socket=DEBUG
logging.level.org.springframework.messaging=DEBUG
```

### Frontend (browser console):

```javascript
localStorage.debug = '*';  // Enable all debug logs
```

---

## 🎯 Quick Fix Commands

### Force send a TwoPacket from backend:

```bash
# Use Postman or curl
curl -X POST http://localhost:8080/api/packets \
  -H "Content-Type: application/json" \
  -d '{
    "dijkstraPacket": {...},
    "rlPacket": {...}
  }'
```

### Clear MongoDB two_packets collection:

```javascript
// MongoDB Compass or mongosh
use network
db.two_packets.deleteMany({})
```

### Reset frontend state:

```javascript
// Browser console
localStorage.clear();
sessionStorage.clear();
location.reload();
```

---

## 🔗 Related Documents

- [CHANGESTREAM_FIX_SUMMARY.md](CHANGESTREAM_FIX_SUMMARY.md)
- [FRONTEND_BACKEND_SYNC_FIX.md](FRONTEND_BACKEND_SYNC_FIX.md)
- [MONGODB_TIMEOUT_FIX.md](MONGODB_TIMEOUT_FIX.md)
- [CHANGESTREAM_DEBUG_GUIDE.md](CHANGESTREAM_DEBUG_GUIDE.md)

---

## ✅ Success Checklist

- [ ] Backend started và logs show "✅ MessageListenerContainer is RUNNING"
- [ ] MongoDB Change Stream nhận được events
- [ ] TwoPacket có cả dijkstraPacket VÀ rlPacket
- [ ] Backend gửi "/topic/packets" message
- [ ] Frontend WebSocket connected
- [ ] Browser console logs "📩 Packet message received"
- [ ] Monitor page hiển thị data
- [ ] PacketRouteGraph và charts render correctly

# Frontend-Backend Data Model Sync - Fix Summary

## 🔍 Vấn đề

Frontend không hiển thị đúng dữ liệu từ backend do **sự không đồng nhất** giữa data models.

---

## ❌ Các vấn đề đã phát hiện

### 1. **Field Name Mismatch: `useRL` vs `isUseRL`**

**Backend (Java - Packet.java:42):**
```java
private boolean isUseRL = false;
```
- JSON serialize: `{"isUseRL": true/false}`

**Frontend (BEFORE FIX):**

- **ComparisonTypes.ts:**
  ```typescript
  useRL: boolean;  // ❌ WRONG
  ```

- **type.ts (cho usePacketProcessor):**
  ```typescript
  isUseRL: boolean;  // ✅ Already correct
  ```

**Kết quả:**
- `useBatchWebSocket` (dùng ComparisonTypes) nhận data từ backend nhưng **field name không khớp**
- `packet.useRL` luôn `undefined`
- Các components không biết packet nào dùng RL, nào dùng Dijkstra
- Hiển thị sai thuật toán

---

### 2. **Field Name Mismatch: `ttl` vs `TTL`**

**Backend:**
```java
private int TTL;
```

**Frontend (BEFORE FIX):**
```typescript
ttl: number;  // ❌ Lowercase, backend là uppercase
```

---

### 3. **ServiceType không chặt chẽ**

**Backend (ServiceType.java):**
```java
public enum ServiceType {
    VIDEO_STREAM,      // Not "VIDEO_STREAMING"
    AUDIO_CALL,
    IMAGE_TRANSFER,
    TEXT_MESSAGE,
    FILE_TRANSFER,
    VIDEO_STREAMING    // Backward compatibility
}
```

**Frontend (ComparisonTypes.ts BEFORE FIX):**
```typescript
interface QoS {
    serviceType: string;  // ❌ Too loose, no type safety
}
```

**Vấn đề:**
- Không có type safety
- Có thể assign bất kỳ string nào
- Typos không được catch

---

## ✅ Giải pháp đã áp dụng

### Fix 1: ComparisonTypes.ts - Field names

**File:** [src/sagsins-frontend/src/types/ComparisonTypes.ts](src/sagsins-frontend/src/types/ComparisonTypes.ts)

```typescript
export interface Packet {
    // ... other fields ...
    isUseRL: boolean;  // ✅ Fixed: Match backend field name (was: useRL)
    TTL: number;       // ✅ Fixed: Uppercase to match backend (was: ttl)
}
```

### Fix 2: ServiceType union type

**File:** [src/sagsins-frontend/src/types/ComparisonTypes.ts](src/sagsins-frontend/src/types/ComparisonTypes.ts)

```typescript
// ✅ ServiceType matching backend enum
export type ServiceType =
    | "VIDEO_STREAM"
    | "VIDEO_STREAMING"  // Backward compatibility
    | "AUDIO_CALL"
    | "IMAGE_TRANSFER"
    | "TEXT_MESSAGE"
    | "FILE_TRANSFER";

export interface QoS {
    serviceType: ServiceType;  // ✅ Fixed: Use union type instead of string
    defaultPriority: number;
    maxLatencyMs: number;
    maxJitterMs: number;
    minBandwidthMbps: number;
    maxLossRate: number;
}
```

**Benefits:**
- Type safety
- IDE autocomplete
- Compile-time error checking
- Matches backend enum values

### Fix 3: Update all usages của `useRL` → `isUseRL`

#### a. calculateCongestionMap.ts

**File:** [src/sagsins-frontend/src/utils/calculateCongestionMap.ts:55](src/sagsins-frontend/src/utils/calculateCongestionMap.ts#L55)

```typescript
// BEFORE:
const isRL = packet.useRL;  // ❌ undefined

// AFTER:
const isRL = packet.isUseRL;  // ✅ Correct
```

#### b. PacketFlowDetail.tsx

**File:** [src/sagsins-frontend/src/components/batchchart/PacketFlowDetail.tsx:30](src/sagsins-frontend/src/components/batchchart/PacketFlowDetail.tsx#L30)

```typescript
// BEFORE:
algorithm: packet.useRL ? 'RL' : 'Dijkstra',  // ❌

// AFTER:
algorithm: packet.isUseRL ? 'RL' : 'Dijkstra',  // ✅
```

#### c. NodeCongestionCard.tsx

**File:** [src/sagsins-frontend/src/components/batchchart/NodeCongestionCard.tsx:86](src/sagsins-frontend/src/components/batchchart/NodeCongestionCard.tsx#L86)

```typescript
// BEFORE:
algorithm: packet.useRL ? 'RL' : 'Dijkstra',  // ❌

// AFTER:
algorithm: packet.isUseRL ? 'RL' : 'Dijkstra',  // ✅
```

#### d. CombinedHopMetricsChart.tsx

**File:** [src/sagsins-frontend/src/components/chart/CombinedHopMetricsChart.tsx:83](src/sagsins-frontend/src/components/chart/CombinedHopMetricsChart.tsx#L83)

Local interface definition:
```typescript
// BEFORE:
interface Packet {
    useRL: boolean;
    ttl: number;
}

// AFTER:
interface Packet {
    isUseRL: boolean;  // ✅ Fixed
    TTL: number;       // ✅ Fixed
}
```

#### e. PacketRouteGraph.tsx

**File:** [src/sagsins-frontend/src/components/chart/PacketRouteGraph.tsx:74](src/sagsins-frontend/src/components/chart/PacketRouteGraph.tsx#L74)

Local interface definition:
```typescript
// BEFORE:
interface Packet {
    useRL: boolean;
    ttl: number;
}

// AFTER:
interface Packet {
    isUseRL: boolean;  // ✅ Fixed
    TTL: number;       // ✅ Fixed
}
```

---

## 📊 Impact Analysis

### Before Fix:

```javascript
// Backend sends:
{
  "isUseRL": true,
  "TTL": 10
}

// Frontend ComparisonTypes expects:
{
  "useRL": boolean,  // ❌ undefined (field doesn't exist in backend response)
  "ttl": number      // ❌ undefined
}

// Result:
packet.useRL === undefined  // ❌ Always falsy
→ All packets shown as "Dijkstra"
→ RL algorithm metrics = 0
→ Incorrect visualization
```

### After Fix:

```javascript
// Backend sends:
{
  "isUseRL": true,
  "TTL": 10
}

// Frontend ComparisonTypes:
{
  "isUseRL": boolean,  // ✅ Matches!
  "TTL": number        // ✅ Matches!
}

// Result:
packet.isUseRL === true  // ✅ Correct value
→ Packets correctly labeled as "RL" or "Dijkstra"
→ Metrics calculated correctly
→ Correct visualization
```

---

## 🧪 Testing

### Manual Testing Steps:

1. **Start Backend:**
   ```bash
   cd src/SAGSINs
   ./mvnw spring-boot:run
   ```

2. **Start Frontend:**
   ```bash
   cd src/sagsins-frontend
   npm start
   ```

3. **Trigger Packet Flow:**
   - Run Python rl-router to send packets
   - Or use test script: `python3 test_change_stream.py`

4. **Verify in Frontend:**

   **Dashboard should show:**
   - ✅ Both "RL" and "Dijkstra" packets (not all Dijkstra)
   - ✅ Correct packet counts for each algorithm
   - ✅ Proper visualization in Sankey diagrams
   - ✅ Accurate metrics for both algorithms

   **Node Congestion should show:**
   - ✅ Correct algorithm labels on each packet
   - ✅ Proper algorithm distribution (dijkstra count vs rl count)

### Browser Console Checks:

```javascript
// In browser console, inspect received batch:
console.log(receivedBatches[0].packets[0]);

// Should see:
{
  dijkstraPacket: {
    isUseRL: false,  // ✅ Correct
    TTL: 10,         // ✅ Correct
    // ...
  },
  rlPacket: {
    isUseRL: true,   // ✅ Correct
    TTL: 10,         // ✅ Correct
    // ...
  }
}
```

---

## 📁 Files Changed

### Type Definitions:
1. **[src/sagsins-frontend/src/types/ComparisonTypes.ts](src/sagsins-frontend/src/types/ComparisonTypes.ts)**
   - ✅ Fixed `useRL` → `isUseRL`
   - ✅ Fixed `ttl` → `TTL`
   - ✅ Added ServiceType union type
   - ✅ Updated QoS interface

### Utilities:
2. **[src/sagsins-frontend/src/utils/calculateCongestionMap.ts](src/sagsins-frontend/src/utils/calculateCongestionMap.ts)**
   - ✅ Fixed packet.useRL → packet.isUseRL

### Components:
3. **[src/sagsins-frontend/src/components/batchchart/PacketFlowDetail.tsx](src/sagsins-frontend/src/components/batchchart/PacketFlowDetail.tsx)**
   - ✅ Fixed packet.useRL → packet.isUseRL

4. **[src/sagsins-frontend/src/components/batchchart/NodeCongestionCard.tsx](src/sagsins-frontend/src/components/batchchart/NodeCongestionCard.tsx)**
   - ✅ Fixed packet.useRL → packet.isUseRL

5. **[src/sagsins-frontend/src/components/chart/CombinedHopMetricsChart.tsx](src/sagsins-frontend/src/components/chart/CombinedHopMetricsChart.tsx)**
   - ✅ Fixed local interface: useRL → isUseRL, ttl → TTL

6. **[src/sagsins-frontend/src/components/chart/PacketRouteGraph.tsx](src/sagsins-frontend/src/components/chart/PacketRouteGraph.tsx)**
   - ✅ Fixed local interface: useRL → isUseRL, ttl → TTL

---

## 🎯 Tóm tắt

### Vấn đề:
- Frontend và Backend có data model không đồng nhất
- Field names không khớp (`useRL` vs `isUseRL`, `ttl` vs `TTL`)
- ServiceType không có type safety

### Giải pháp:
- ✅ Sync tất cả field names với backend
- ✅ Thêm ServiceType union type
- ✅ Update tất cả usages trong code

### Kết quả:
- ✅ Frontend nhận và parse đúng data từ backend
- ✅ Hiển thị đúng thuật toán cho mỗi packet
- ✅ Metrics và visualization chính xác
- ✅ Type safety improved

---

## 🔗 Related Documents

- [CHANGESTREAM_FIX_SUMMARY.md](CHANGESTREAM_FIX_SUMMARY.md) - Enum mismatch fix
- [MONGODB_TIMEOUT_FIX.md](MONGODB_TIMEOUT_FIX.md) - Connection timeout fix
- [CHANGESTREAM_DEBUG_GUIDE.md](CHANGESTREAM_DEBUG_GUIDE.md) - Debug guide

---

## 💡 Best Practices Moving Forward

1. **Keep types in sync:**
   - Document backend data models
   - Generate TypeScript types from OpenAPI/Swagger
   - Or use shared type definitions

2. **Use strict TypeScript:**
   - Enable `strictNullChecks`
   - Use union types instead of `string`
   - Avoid `any` type

3. **Runtime validation:**
   - Consider using Zod or Yup for runtime type checking
   - Validate API responses match expected schema

4. **Testing:**
   - Add integration tests that verify data flow
   - Mock API responses with real backend data
   - Test edge cases (null values, missing fields)

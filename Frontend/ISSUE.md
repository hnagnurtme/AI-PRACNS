# ISSUE.md - Frontend Implementation

## Đề tài: Tối ưu hóa phân bổ tài nguyên trong mạng SAGIN sử dụng Reinforcement Learning

### 📋 Tổng quan
Frontend React + TypeScript cung cấp giao diện người dùng để:
- Hiển thị mạng SAGIN 3D trên Cesium
- Quản lý và theo dõi nodes (Satellite, Ground Station)
- Mô phỏng và so sánh routing algorithms (Dijkstra vs RL)
- Hiển thị real-time metrics qua WebSocket
- Phân tích batch packets và network topology

---

## 🎯 Phase 1: Thiết lập Routing System

### 1.1. Cài đặt React Router
- [ ] **Cài đặt dependencies**
  - [ ] `npm install react-router-dom`
  - [ ] Cài đặt types: `npm install --save-dev @types/react-router-dom` (nếu cần)

### 1.2. Cấu trúc Routing
- [ ] **src/App.tsx**
  - [ ] Thay thế state-based navigation bằng React Router
  - [ ] Setup `BrowserRouter` hoặc `HashRouter`
  - [ ] Định nghĩa routes:
    - [ ] `/` hoặc `/dashboard` - Page 1: Dashboard với Cesium 3D
    - [ ] `/topology` - Page 2: Tổng quan topology networks
    - [ ] `/monitor` - Page 3: Kịch bản mô phỏng và biểu đồ so sánh
    - [ ] `/comparison` - Page 4: Hiển thị 2 gói tin (Dijkstra vs RL)
    - [ ] `/batch` - Page 5: Batch gói tin monitor
  - [ ] Setup 404 page (Not Found)

- [ ] **src/layouts/HeaderLayout.tsx**
  - [ ] Thay thế `activePage` state bằng `useNavigate` và `useLocation`
  - [ ] Update navigation buttons để sử dụng `navigate()` thay vì `setActivePage()`
  - [ ] Highlight active route dựa trên `location.pathname`
  - [ ] Update `PageName` type để match với routes

- [ ] **src/layouts/MainLayout.tsx**
  - [ ] Remove `activePage` prop
  - [ ] Wrap với `Outlet` từ React Router
  - [ ] Giữ nguyên Header và Footer

---

## 🗺️ Phase 2: Page 1 - Dashboard (Cesium 3D với User Terminals)

### 2.1. Generate User Terminals
- [ ] **src/utils/userTerminalGenerator.ts**
  - [ ] `generateRandomUserTerminals(count: number, bounds: Bounds)` - Generate ngẫu nhiên
    - [ ] Random latitude/longitude trong bounds
    - [ ] Random altitude (ground level: 0-100m)
    - [ ] Generate unique terminal IDs
    - [ ] Assign random QoS requirements
  - [ ] `generateUserTerminalsInRegion(region: Region, density: number)` - Generate theo region
  - [ ] `validateTerminalPosition(terminal: UserTerminal, nodes: NodeDTO[])` - Validate không trùng với nodes

- [ ] **src/types/UserTerminal.ts**
  - [ ] Định nghĩa `UserTerminal` interface
    - [ ] `terminalId: string`
    - [ ] `position: Position` (lat, lon, alt)
    - [ ] `qosRequirements: QoS`
    - [ ] `connectedNodeId?: string` (node đang kết nối)
    - [ ] `status: 'idle' | 'connected' | 'transmitting'`
    - [ ] `metadata: { name, type, etc }`

### 2.2. API Integration cho User Terminals
- [ ] **src/api/userTerminalApi.ts**
  - [ ] `generateUserTerminalsRequest(count, bounds)` - Gửi request generate
  - [ ] `getUserTerminals()` - Lấy danh sách terminals
  - [ ] `connectTerminalToNode(terminalId, nodeId)` - Kết nối terminal với node
  - [ ] `getTerminalConnectionResult(terminalId)` - Lấy kết quả kết nối

- [ ] **src/hooks/useUserTerminals.ts**
  - [ ] Hook để quản lý user terminals
  - [ ] State: terminals, loading, error
  - [ ] Functions: generate, refresh, connect
  - [ ] Auto-refresh khi có updates từ WebSocket

### 2.3. WebSocket Integration cho Connection Results
- [ ] **src/hooks/useTerminalWebSocket.ts**
  - [ ] Subscribe to `/topic/terminal-connections`
  - [ ] Nhận real-time connection results
  - [ ] Update terminal status
  - [ ] Trigger re-render khi có updates

- [ ] **src/contexts/WebSocketContext.tsx** (extend existing)
  - [ ] Thêm subscription cho terminal updates
  - [ ] Broadcast terminal connection results

### 2.4. Cesium Visualization cho User Terminals
- [ ] **src/map/CesiumViewer.tsx** (extend existing)
  - [ ] Thêm visualization cho user terminals
    - [ ] Billboard icons cho terminals (khác với nodes)
    - [ ] Labels với terminal ID
    - [ ] Color coding theo status (idle/connected/transmitting)
  - [ ] Vẽ connection lines giữa terminal và node
    - [ ] Polyline từ terminal đến connected node
    - [ ] Animate connection khi establish
    - [ ] Update line khi connection changes
  - [ ] Click handler cho terminals
    - [ ] Show terminal detail card
    - [ ] Highlight connected node

### 2.5. Terminal Detail Card
- [ ] **src/components/terminals/TerminalDetailCard.tsx**
  - [ ] Hiển thị thông tin terminal
    - [ ] Terminal ID, position
    - [ ] Status (idle/connected/transmitting)
    - [ ] Connected node info
    - [ ] QoS requirements
    - [ ] Connection metrics (latency, bandwidth, etc.)
  - [ ] Actions:
    - [ ] Connect/Disconnect button
    - [ ] Fly to terminal trên map
    - [ ] Show connection path

### 2.6. Dashboard Page Updates
- [ ] **src/pages/Dashboard.tsx**
  - [ ] Thêm controls để generate user terminals
    - [ ] Input số lượng terminals
    - [ ] Select region/bounds
    - [ ] Generate button
  - [ ] Hiển thị list terminals (sidebar hoặc panel)
  - [ ] Filter terminals theo status
  - [ ] Display connection results summary
  - [ ] Integrate với CesiumViewer để hiển thị terminals
  - [ ] Show TerminalDetailCard khi click terminal

---

## 🌐 Phase 3: Page 2 - Network Topology Overview

### 3.1. Network Topology API
- [ ] **src/api/networkTopologyApi.ts**
  - [ ] `getAllNetworks()` - Lấy danh sách tất cả networks
  - [ ] `getNetworkTopology(networkId)` - Lấy topology của network
  - [ ] `getNetworkStatistics(networkId)` - Thống kê network
  - [ ] `getNetworkConnections(networkId)` - Lấy connections giữa nodes

- [ ] **src/types/NetworkTopology.ts**
  - [ ] `Network` interface
    - [ ] `networkId: string`
    - [ ] `name: string`
    - [ ] `nodes: NodeDTO[]`
    - [ ] `connections: Connection[]`
    - [ ] `statistics: NetworkStatistics`
  - [ ] `Connection` interface
    - [ ] `fromNodeId: string`
    - [ ] `toNodeId: string`
    - [ ] `latency: number`
    - [ ] `bandwidth: number`
    - [ ] `status: 'active' | 'inactive' | 'degraded'`
  - [ ] `NetworkStatistics` interface
    - [ ] `totalNodes: number`
    - [ ] `activeConnections: number`
    - [ ] `totalBandwidth: number`
    - [ ] `averageLatency: number`

### 3.2. WebSocket cho Network Topology
- [ ] **src/hooks/useNetworkTopologyWebSocket.ts**
  - [ ] Subscribe to `/topic/network-topology`
  - [ ] Nhận real-time topology updates
  - [ ] Update network connections status
  - [ ] Update node states trong topology

- [ ] **src/contexts/WebSocketContext.tsx** (extend existing)
  - [ ] Thêm subscription cho network topology updates
  - [ ] Broadcast topology changes

### 3.3. Topology Visualization Component
- [ ] **src/components/topology/NetworkTopologyView.tsx**
  - [ ] Hiển thị danh sách networks
    - [ ] Network cards với statistics
    - [ ] Filter và search networks
    - [ ] Select network để xem chi tiết
  - [ ] Topology graph visualization
    - [ ] Sử dụng library như `react-force-graph` hoặc `vis-network`
    - [ ] Nodes (satellites, stations) với icons
    - [ ] Edges (connections) với weights (latency, bandwidth)
    - [ ] Color coding theo status
    - [ ] Interactive: zoom, pan, select node
  - [ ] Network statistics panel
    - [ ] Total nodes, connections
    - [ ] Average latency, bandwidth utilization
    - [ ] Health status

- [ ] **src/components/topology/NetworkCard.tsx**
  - [ ] Card hiển thị network info
  - [ ] Quick stats
  - [ ] Click để xem chi tiết

- [ ] **src/components/topology/ConnectionLine.tsx**
  - [ ] Component vẽ connection line
  - [ ] Animate khi có traffic
  - [ ] Tooltip với connection details

### 3.4. Topology Page
- [ ] **src/pages/Topology.tsx** (new file)
  - [ ] Layout với network list và topology graph
  - [ ] Connect WebSocket cho real-time updates
  - [ ] Filter và search functionality
  - [ ] Network selection và detail view
  - [ ] Export topology (JSON/image)

---

## 📊 Phase 4: Page 3 - Monitor (Kịch bản mô phỏng và Biểu đồ)

### 4.1. Scenario Management
- [ ] **src/components/simulation/ScenarioSelector.tsx** (enhance existing)
  - [ ] Thêm scenario configuration
    - [ ] Network load scenarios
    - [ ] Failure scenarios (node down, link failure)
    - [ ] Traffic patterns
  - [ ] Start/Stop simulation controls
  - [ ] Scenario parameters input

- [ ] **src/api/simulationApi.ts**
  - [ ] `startSimulation(scenarioConfig)` - Bắt đầu simulation
  - [ ] `stopSimulation(simulationId)` - Dừng simulation
  - [ ] `getSimulationStatus(simulationId)` - Trạng thái simulation
  - [ ] `getSimulationResults(simulationId)` - Kết quả simulation

### 4.2. WebSocket cho Simulation Results
- [ ] **src/hooks/useSimulationWebSocket.ts**
  - [ ] Subscribe to `/topic/simulation-results`
  - [ ] Nhận real-time simulation metrics
  - [ ] Update charts với data mới

### 4.3. Comparison Charts (enhance existing)
- [ ] **src/components/chart/CombinedHopMetricsChart.tsx** (enhance existing)
  - [ ] Thêm comparison giữa Dijkstra và RL
  - [ ] Side-by-side metrics
  - [ ] Time-series data
  - [ ] Interactive tooltips

- [ ] **src/components/chart/PacketRouteGraph.tsx** (enhance existing)
  - [ ] Support cho 2 routes (Dijkstra vs RL)
  - [ ] Color coding cho mỗi algorithm
  - [ ] Highlight differences

- [ ] **src/components/chart/AlgorithmComparisonChart.tsx** (new hoặc enhance)
  - [ ] So sánh performance metrics
    - [ ] Latency comparison
    - [ ] Success rate
    - [ ] Resource utilization
  - [ ] Bar charts, line charts
  - [ ] Statistical summary

### 4.4. Monitor Page Updates
- [ ] **src/pages/Monitor.tsx** (enhance existing)
  - [ ] Scenario selector với start/stop
  - [ ] Real-time metrics display
  - [ ] Comparison charts (Dijkstra vs RL)
  - [ ] Export results functionality
  - [ ] Historical data view

---

## 🔄 Phase 5: Page 4 - Packet Comparison (2 Gói tin)

### 5.1. Packet Comparison API
- [ ] **src/api/packetComparisonApi.ts**
  - [ ] `getPacketPair(packetId1, packetId2)` - Lấy 2 packets để so sánh
  - [ ] `getComparisonMetrics(packetId1, packetId2)` - Metrics so sánh
  - [ ] `getPacketRoute(packetId)` - Route của packet

### 5.2. WebSocket cho Packet Comparison
- [ ] **src/hooks/usePacketComparisonWebSocket.ts**
  - [ ] Subscribe to `/topic/packet-comparison`
  - [ ] Nhận real-time packet pairs
  - [ ] Update comparison view

### 5.3. Comparison Components
- [ ] **src/components/comparison/PacketComparisonView.tsx**
  - [ ] Side-by-side display 2 packets
    - [ ] Left: Dijkstra packet
    - [ ] Right: RL packet
  - [ ] Route visualization cho mỗi packet
  - [ ] Metrics comparison table
  - [ ] Highlight differences

- [ ] **src/components/comparison/PacketDetailPanel.tsx**
  - [ ] Chi tiết packet (Dijkstra hoặc RL)
    - [ ] Packet info (ID, source, destination)
    - [ ] Route path với hops
    - [ ] Metrics (latency, distance, success)
    - [ ] Timeline visualization

- [ ] **src/components/comparison/ComparisonMetricsTable.tsx**
  - [ ] Table so sánh metrics
    - [ ] Latency: Dijkstra vs RL
    - [ ] Distance: Dijkstra vs RL
    - [ ] Hop count: Dijkstra vs RL
    - [ ] Success rate
  - [ ] Highlight winner cho mỗi metric
  - [ ] Statistical significance indicators

- [ ] **src/components/comparison/RouteComparisonMap.tsx**
  - [ ] Map hiển thị 2 routes
  - [ ] Overlay trên Cesium hoặc 2D map
  - [ ] Color coding cho mỗi route
  - [ ] Animate packet movement

### 5.4. Comparison Page
- [ ] **src/pages/Comparison.tsx** (new file)
  - [ ] Layout với 2 panels (Dijkstra vs RL)
  - [ ] Packet selector (chọn packet pair)
  - [ ] Real-time updates từ WebSocket
  - [ ] Export comparison report
  - [ ] Historical comparison view

---

## 📦 Phase 6: Page 5 - Batch Monitor (enhance existing)

### 6.1. Batch WebSocket (enhance existing)
- [ ] **src/hooks/useBatchWebSocket.ts** (enhance existing)
  - [ ] Đảm bảo nhận đúng batch data
  - [ ] Handle reconnection
  - [ ] Buffer management cho large batches

### 6.2. Batch Components (enhance existing)
- [ ] **src/components/batchchart/BatchStatistics.tsx** (enhance existing)
  - [ ] Thêm comparison metrics
  - [ ] Real-time updates
  - [ ] Export functionality

- [ ] **src/components/batchchart/NetworkTopologyView.tsx** (enhance existing)
  - [ ] Improve visualization
  - [ ] Better congestion display
  - [ ] Interactive node selection

- [ ] **src/components/batchchart/PacketFlowDetail.tsx** (enhance existing)
  - [ ] More detailed packet flow
  - [ ] Timeline visualization
  - [ ] Filter và search

- [ ] **src/components/batchchart/AlgorithmComparisonChart.tsx** (enhance existing)
  - [ ] Better comparison visualization
  - [ ] Statistical analysis
  - [ ] Export charts

### 6.3. Batch Monitor Page (enhance existing)
- [ ] **src/pages/BatchMonitor.tsx** (enhance existing)
  - [ ] Improve layout và UX
  - [ ] Better error handling
  - [ ] Loading states
  - [ ] Export batch results

---

## 🔌 Phase 7: WebSocket Infrastructure Enhancement

### 7.1. Centralized WebSocket Management
- [ ] **src/contexts/WebSocketContext.tsx** (enhance existing)
  - [ ] Support multiple subscriptions
  - [ ] Topic management
  - [ ] Reconnection logic improvement
  - [ ] Error handling và retry
  - [ ] Connection status indicator

- [ ] **src/hooks/useWebSocket.ts** (generic hook)
  - [ ] Generic hook cho WebSocket subscriptions
  - [ ] Auto-reconnect
  - [ ] Message buffering
  - [ ] Error handling

### 7.2. WebSocket Topics
- [ ] Định nghĩa tất cả WebSocket topics:
  - [ ] `/topic/node-status` - Node status updates (existing)
  - [ ] `/topic/packets` - Packet updates (existing)
  - [ ] `/topic/batch-packets` - Batch packet updates (existing)
  - [ ] `/topic/terminal-connections` - Terminal connection results (new)
  - [ ] `/topic/network-topology` - Network topology updates (new)
  - [ ] `/topic/simulation-results` - Simulation results (new)
  - [ ] `/topic/packet-comparison` - Packet comparison updates (new)

---

## 🎨 Phase 8: UI/UX Improvements

### 8.1. Loading States
- [ ] **src/components/common/LoadingSpinner.tsx**
  - [ ] Reusable loading component
  - [ ] Different sizes và styles

- [ ] **src/components/common/ErrorBoundary.tsx**
  - [ ] Error boundary cho error handling
  - [ ] User-friendly error messages

### 8.2. Notifications
- [ ] **src/components/common/Notification.tsx**
  - [ ] Toast notifications
  - [ ] Success/Error/Info/Warning types
  - [ ] Auto-dismiss

- [ ] **src/hooks/useNotification.ts**
  - [ ] Hook để show notifications
  - [ ] Queue management

### 8.3. Responsive Design
- [ ] Ensure all pages responsive
- [ ] Mobile-friendly layouts
- [ ] Touch interactions cho mobile

---

## 🧪 Phase 9: Testing

### 9.1. Unit Tests
- [ ] **src/utils/userTerminalGenerator.test.ts**
  - [ ] Test terminal generation
  - [ ] Test validation

- [ ] **src/hooks/useUserTerminals.test.ts**
  - [ ] Test hook functionality

- [ ] **src/components/** - Component tests

### 9.2. Integration Tests
- [ ] **src/pages/** - Page integration tests
- [ ] **WebSocket integration tests**
- [ ] **API integration tests**

### 9.3. E2E Tests
- [ ] **cypress/** hoặc **playwright/**
  - [ ] Navigation tests
  - [ ] User interactions
  - [ ] WebSocket connections

---

## 📚 Phase 10: Documentation & Code Quality

### 10.1. Code Documentation
- [ ] JSDoc comments cho tất cả functions/components
- [ ] Type definitions đầy đủ
- [ ] README updates

### 10.2. Type Safety
- [ ] Ensure all types defined
- [ ] No `any` types
- [ ] Strict TypeScript mode

### 10.3. Performance Optimization
- [ ] React.memo cho expensive components
- [ ] useMemo/useCallback optimization
- [ ] Code splitting với React.lazy
- [ ] Bundle size optimization

---

## ✅ Checklist tổng hợp

### Routing
- [ ] React Router setup
- [ ] 5 pages với routes
- [ ] Navigation updates
- [ ] 404 page

### Page 1 - Dashboard
- [ ] User terminal generator
- [ ] Terminal visualization trên Cesium
- [ ] Connection results display
- [ ] Terminal detail card
- [ ] WebSocket integration

### Page 2 - Topology
- [ ] Network topology API
- [ ] Topology visualization
- [ ] WebSocket cho real-time updates
- [ ] Network statistics

### Page 3 - Monitor
- [ ] Scenario management
- [ ] Simulation controls
- [ ] Comparison charts
- [ ] WebSocket integration

### Page 4 - Comparison
- [ ] Packet comparison view
- [ ] Side-by-side display
- [ ] Comparison metrics
- [ ] Route visualization

### Page 5 - Batch Monitor
- [ ] Enhance existing components
- [ ] Better visualization
- [ ] Export functionality

### Infrastructure
- [ ] WebSocket enhancements
- [ ] Error handling
- [ ] Loading states
- [ ] Notifications

### Quality
- [ ] Tests
- [ ] Documentation
- [ ] Performance optimization

---

## 🎯 Priority Order

1. **Phase 1**: Routing setup (Foundation)
2. **Phase 2**: Page 1 - Dashboard với user terminals
3. **Phase 7**: WebSocket infrastructure (cần cho các pages)
4. **Phase 3**: Page 2 - Topology
5. **Phase 4**: Page 3 - Monitor enhancements
6. **Phase 5**: Page 4 - Comparison
7. **Phase 6**: Page 5 - Batch Monitor enhancements
8. **Phase 8-10**: UI/UX, Testing, Documentation

---

## 📝 Notes

- Sử dụng React Router v6
- WebSocket: STOMP over SockJS (đã có)
- Cesium cho 3D visualization (đã có)
- Recharts hoặc Chart.js cho charts (kiểm tra existing)
- Zustand cho state management (đã có nodeStore)
- TypeScript strict mode
- Responsive design với Tailwind CSS (đã có)

---

## 🔗 Dependencies cần thêm

```json
{
  "react-router-dom": "^6.x",
  "react-force-graph": "^1.x" // hoặc vis-network cho topology graph
}
```


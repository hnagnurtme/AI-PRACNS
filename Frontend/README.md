# SAGIN Frontend - 3D Network Visualization

A modern **React + TypeScript** frontend application for visualizing and monitoring Space-Air-Ground Integrated Networks (SAGIN) with real-time routing and 3D globe visualization using Cesium.

---

## 🚀 Features

### 3D Visualization
- **Cesium 3D Globe** - Interactive 3D visualization of Earth with satellite orbits
- **Real-time Network Topology** - Dynamic visualization of satellites, ground stations, and UAVs
- **Routing Path Animation** - Real-time packet routing visualization with animated paths
- **Node Information** - Interactive node details with status and metrics

### Network Monitoring
- **Real-time Metrics** - Live network performance monitoring
- **Performance Comparison** - Compare RL agent vs traditional algorithms (Dijkstra, Heuristic)
- **Historical Data** - Track network performance over time with charts
- **Alert System** - Real-time notifications for network events

### Routing Management
- **Manual Routing** - Select source and destination for packet transmission
- **Algorithm Selection** - Choose between RL Agent, Dijkstra, or Heuristic routing
- **Path Visualization** - View complete routing paths with hop-by-hop details
- **Performance Metrics** - Latency, hops, distance, and success rate analysis

### Guest Access
- **Demo Mode** - Explore the system without authentication
- **Interactive Tutorials** - Step-by-step guides for new users
- **Limited Features** - Safe exploration environment

---

## 🛠️ Technology Stack

### Core
- **React 19** - UI framework
- **TypeScript** - Type-safe development
- **Vite 5** - Fast build tool and dev server

### 3D Visualization
- **Cesium** - 3D globe and geospatial visualization
- **React Force Graph** - Network graph visualization

### UI/UX
- **TailwindCSS 3** - Utility-first CSS framework
- **Lucide React** - Icon library
- **React Toastify** - Toast notifications
- **React Draggable** - Draggable UI components

### Data Visualization
- **Recharts** - Chart library for performance metrics

### Real-time Communication
- **Socket.IO Client** - WebSocket for real-time updates
- **STOMP.js** - STOMP protocol over WebSocket
- **Axios** - HTTP client for API calls

---

## 📦 Installation

### Prerequisites
- Node.js >= 16.x
- npm or yarn

### Setup

```bash
# Navigate to Frontend directory
cd Frontend

# Install dependencies
npm install

# Copy Cesium assets (automatically runs after install)
npm run copy-cesium

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the Frontend directory:

```env
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080
```

### Cesium Configuration

Cesium assets are automatically copied to `public/Cesium/` during installation. If you need to manually copy them:

```bash
npm run copy-cesium
```

---

## 🎯 Available Scripts

### Development

```bash
# Start development server with hot reload
npm run dev

# Alternative start command
npm start
```

### Build

```bash
# Type check and build for production
npm run build

# Preview production build
npm run preview
```

### Code Quality

```bash
# Lint code
npm run lint
```

---

## 📂 Project Structure

```
Frontend/
├── public/
│   ├── Cesium/              # Cesium assets (auto-generated)
│   └── icons/               # App icons and assets
├── src/
│   ├── api/                 # API client and services
│   ├── assets/              # Static assets
│   ├── components/          # React components
│   │   ├── cesium/          # Cesium-related components
│   │   ├── charts/          # Chart components
│   │   ├── routing/         # Routing components
│   │   └── ui/              # UI components
│   ├── contexts/            # React contexts
│   ├── hooks/               # Custom React hooks
│   ├── layouts/             # Layout components
│   ├── map/                 # Map-related utilities
│   ├── pages/               # Page components
│   ├── services/            # Business logic services
│   ├── state/               # State management
│   ├── types/               # TypeScript types
│   ├── utils/               # Utility functions
│   ├── App.tsx              # Main App component
│   └── main.tsx             # Entry point
├── scripts/
│   └── copy-cesium.cjs      # Script to copy Cesium assets
├── index.html               # HTML template
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind configuration
└── tsconfig.json            # TypeScript configuration
```

---

## 🎨 Key Components

### CesiumGlobe
3D globe visualization with satellite tracking and network topology

```typescript
import { CesiumGlobe } from '@/components/cesium/CesiumGlobe';

<CesiumGlobe 
  nodes={nodes}
  terminals={terminals}
  paths={routingPaths}
/>
```

### RoutingControl
Control panel for packet routing

```typescript
import { RoutingControl } from '@/components/routing/RoutingControl';

<RoutingControl 
  onRoute={handleRoute}
  algorithms={['rl_agent', 'dijkstra', 'heuristic']}
/>
```

### PerformanceChart
Performance metrics visualization

```typescript
import { PerformanceChart } from '@/components/charts/PerformanceChart';

<PerformanceChart 
  data={performanceData}
  metrics={['latency', 'hops', 'distance']}
/>
```

---

## 🌐 API Integration

### REST API

```typescript
import { api } from '@/api/client';

// Get network nodes
const nodes = await api.get('/api/nodes');

// Send routing request
const result = await api.post('/api/routing/route', {
  source: 'terminal_1',
  destination: 'terminal_2',
  algorithm: 'rl_agent'
});
```

### WebSocket

```typescript
import { socket } from '@/services/websocket';

// Subscribe to real-time updates
socket.on('network_update', (data) => {
  console.log('Network update:', data);
});

// Subscribe to routing events
socket.on('routing_update', (data) => {
  console.log('Routing update:', data);
});
```

---

## 🎯 Features in Detail

### Network Topology Visualization
- Real-time 3D visualization of satellites, ground stations, and UAVs
- Orbital paths and coverage areas
- Interactive node selection and information display
- Dynamic link status (active/inactive)

### Routing Visualization
- Animated packet transmission along routing paths
- Hop-by-hop visualization with timing
- Multiple simultaneous route display
- Path comparison between algorithms

### Performance Monitoring
- Real-time latency, throughput, and packet loss metrics
- Historical performance charts
- Algorithm comparison dashboard
- Success rate and reliability tracking

### User Interactions
- Click nodes for detailed information
- Drag and zoom 3D globe
- Select terminals for routing
- Switch between visualization modes (3D globe, 2D network graph)

---

## 🐛 Troubleshooting

### Cesium Assets Not Loading
```bash
# Manually copy Cesium assets
npm run copy-cesium

# Check if Cesium folder exists in public/
ls -la public/Cesium
```

### WebSocket Connection Issues
- Check backend server is running
- Verify `VITE_WS_URL` in `.env`
- Check CORS settings on backend

### Build Errors
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

### Performance Issues
- Reduce number of displayed satellites in settings
- Disable particle effects
- Use Chrome for better WebGL performance

---

## 🚀 Deployment

### Production Build

```bash
# Build for production
npm run build

# Preview production build locally
npm run preview
```

The build output will be in the `dist/` directory.

### Docker Deployment

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment-specific Builds

```bash
# Development
npm run build -- --mode development

# Production
npm run build -- --mode production
```

---

## 📚 Documentation

- [React Documentation](https://react.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Vite Documentation](https://vitejs.dev/)
- [Cesium Documentation](https://cesium.com/docs/)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)

---

## 🤝 Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Write meaningful commit messages
4. Test your changes thoroughly
5. Update documentation as needed

---

## 📝 License

This project is part of the AI-PRACNS system.

---

**Built with ❤️ by TheElite Team**

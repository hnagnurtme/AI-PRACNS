# Components Structure

## 📁 Layout Components (`/layout/`)
- **Layout.tsx** - Main layout component that orchestrates the entire UI
- **Topbar.tsx** - Top navigation bar with logo, status, and user controls
- **Sidebar.tsx** - Left sidebar displaying satellite nodes list

## 🛰️ Cesium Components (`/cesium/`)
- **CesiumMap.tsx** - Main 3D globe component with Cesium integration

## 🎨 UI Components (`/ui/`)
- **LoadingScreen.tsx** - Loading state component
- **ErrorScreen.tsx** - Error state component

## 📋 Features

### Layout System
- **Responsive Design**: Sidebar can be toggled on/off
- **Clean Separation**: Layout logic separated from business logic
- **State Management**: Centralized state for sidebar and node selection

### Sidebar Features
- **Node List**: Displays all satellite nodes with details
- **Node Selection**: Click to select, double-click to focus
- **Node Information**: Shows position, altitude, and type
- **Visual Indicators**: Color-coded status and selection states

### Topbar Features
- **Toggle Sidebar**: Hamburger menu to show/hide sidebar
- **System Status**: Connection status and view mode
- **User Controls**: Settings and user profile
- **Branding**: Logo and system name

### Cesium Integration
- **3D Globe**: Full Cesium 3D globe with terrain
- **Node Visualization**: Satellite positions as 3D points
- **Interactive**: Click nodes to focus, hover for details
- **Default Tools**: All standard Cesium controls enabled
- **Camera Control**: Smooth transitions and focus animations

## 🎯 Usage

```tsx
import Layout from './components/layout/Layout';

// Use in your main component
<Layout 
  nodes={nodes}
  loading={loading}
  error={error}
  onRetry={retry}
/>
```

## 🔧 Configuration

The layout automatically handles:
- Sidebar toggle state
- Node selection and focus
- Loading and error states
- Cesium map initialization
- Event handling and callbacks

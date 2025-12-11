import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import MainLayoutSimple from './layout/MainLayoutSimple';
import MapPage from './pages/MapPage';
import CesiumTestPage from './pages/CesiumTestPage';
import SimulationPageSimple from './pages/SimulationPageSimple';
import DashboardPageSimple from './pages/DashboardPageSimple';
import TestPage from './pages/TestPage';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  console.log('App render');
  
  return (
    <ErrorBoundary>
      <Router>
        <MainLayoutSimple>
          <Routes>
            <Route path="/test" element={<TestPage />} />
            <Route path="/cesium-test" element={<CesiumTestPage />} />
            <Route path="/" element={<MapPage />} />
            <Route path="/simulation" element={<SimulationPageSimple />} />
            <Route path="/dashboard" element={<DashboardPageSimple />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </MainLayoutSimple>
      </Router>
    </ErrorBoundary>
  );
}

export default App;

import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import { ComparisonDashboard } from './pages/Monitor';
import BatchDashboard from './pages/BatchMonitor';
import Topology from './pages/Topology';
import Comparison from './pages/Comparison';

const App: React.FC = () => {
    return (
        <Routes>
            <Route path="/" element={<MainLayout />}>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="topology" element={<Topology />} />
                <Route path="monitor" element={<ComparisonDashboard />} />
                <Route path="comparison" element={<Comparison />} />
                <Route path="batch" element={<BatchDashboard />} />
                <Route path="*" element={
                    <div className="flex items-center justify-center h-screen">
                        <div className="text-center">
                            <h1 className="text-4xl font-bold text-gray-800 mb-4">404</h1>
                            <p className="text-gray-600">Page not found</p>
                        </div>
                    </div>
                } />
            </Route>
        </Routes>
    );
};

export default App;
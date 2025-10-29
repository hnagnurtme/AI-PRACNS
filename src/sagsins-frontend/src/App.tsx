import React from 'react';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import { useState } from 'react';
import type { PageName } from './layouts/HeaderLayout';
import { ComparisonDashboard } from './pages/Monitor';
import BatchDashboard from './pages/BatchMonitor';

const App: React.FC = () => {
    const [activePage, setActivePage] = useState<PageName>('dashboard'); 
    const renderPage = () => {
        switch (activePage) {
            case 'dashboard':
                return <Dashboard />;
            case 'monitor':
                return <ComparisonDashboard />;
            case 'batch':
                return <BatchDashboard />;
            default:
                return <Dashboard />; 
        }
    };

    return (
        <MainLayout activePage={activePage} setActivePage={setActivePage}>
            {/* Render trang con tương ứng */}
            {renderPage()} 
        </MainLayout>
    );
};

export default App;
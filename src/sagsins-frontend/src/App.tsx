import React from 'react';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import Monitor from './pages/Monitor';
import { useState } from 'react';
import type { PageName } from './layouts/HeaderLayout';

const App: React.FC = () => {
    const [activePage, setActivePage] = useState<PageName>('dashboard'); 
    const renderPage = () => {
        switch (activePage) {
            case 'dashboard':
                return <Dashboard />;
            case 'monitor':
                return <Monitor />;
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
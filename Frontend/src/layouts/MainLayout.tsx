// src/components/layout/MainLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './HeaderLayout'; 

const MainLayout: React.FC = () => {
    return (
        <div className="flex flex-col h-full w-screen overflow-hidden bg-gray-100">
            <Header />
            
            <main className="flex-1 overflow-hidden">
                <Outlet />
            </main>
        </div>
    );
};

export default MainLayout;
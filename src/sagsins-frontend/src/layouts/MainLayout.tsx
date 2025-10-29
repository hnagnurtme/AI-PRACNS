// src/components/layout/MainLayout.tsx
import React from 'react';
// [SỬA] Import thêm PageName từ Header
import Header, {type PageName } from './HeaderLayout'; 

interface MainLayoutProps {
    children: React.ReactNode;
    activePage: PageName;
    setActivePage: (page: PageName) => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, activePage, setActivePage }) => {
    return (
        <div className="flex flex-col h-full w-screen overflow-hidden bg-gray-100">
            {/* [SỬA] Truyền props xuống Header */}
            <Header activePage={activePage} setActivePage={setActivePage} />
            
            <main className="flex-1 overflow-hidden">
                {children} 
            </main>
        </div>
    );
};

export default MainLayout;
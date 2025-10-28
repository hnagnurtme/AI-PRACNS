// src/components/layout/MainLayout.tsx
import React from 'react';
// [SỬA] Import thêm PageName từ Header
import Header, {type PageName } from './HeaderLayout'; 
import Footer from './FooterLayout';

interface MainLayoutProps {
    children: React.ReactNode;
    // [SỬA] Thêm props để nhận từ App
    activePage: PageName;
    setActivePage: (page: PageName) => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, activePage, setActivePage }) => {
    return (
        <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-100">
            {/* [SỬA] Truyền props xuống Header */}
            <Header activePage={activePage} setActivePage={setActivePage} />
            
            <main className="flex-1 overflow-hidden">
                {children} 
            </main>
            
            <Footer />
        </div>
    );
};

export default MainLayout;
import React from 'react';

export type PageName = 'dashboard' | 'monitor';

interface HeaderProps {
    activePage: PageName;
    setActivePage: (page: PageName) => void;
}

const Header: React.FC<HeaderProps> = ({ activePage, setActivePage }) => {
    
    const getTabClassName = (pageName: PageName): string => {
        const baseStyle = "px-4 py-2 text-sm font-medium rounded-md transition-colors duration-150";
        if (activePage === pageName) {
            return `${baseStyle} bg-indigo-500 text-white`; 
        }
        return `${baseStyle} text-gray-300 hover:bg-gray-700 hover:text-white`; 
    };

    return (
        <header className="w-full h-16 bg-gray-800 text-white flex items-center p-4 shadow-md z-20">
            {/* --- Logo và Tiêu đề (giữ nguyên) --- */}
            <div className="flex items-center gap-3">
                <svg 
                    className="w-8 h-8" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24" 
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
                <h1 className="text-xl font-bold mr-6">
                    SAGSINs
                </h1>
            </div>

            {/* --- [SỬA] Xóa <nav> khỏi đây --- */}

            {/* --- [SỬA] Đẩy toàn bộ phần bên phải qua ml-auto --- */}
            <div className="ml-auto flex items-center space-x-4">
                
                {/* --- Thanh điều hướng Tab (Đã chuyển vào đây) --- */}
                <nav className="flex items-center space-x-2">
                    <button 
                        className={getTabClassName('dashboard')}
                        onClick={() => setActivePage('dashboard')}
                    >
                        Dashboard
                    </button>
                    <button 
                        className={getTabClassName('monitor')}
                        onClick={() => setActivePage('monitor')}
                    >
                        Monitor
                    </button>
                </nav>

                {/* --- (Tùy chọn) Dấu gạch phân cách cho đẹp --- */}
                <div className="h-6 w-px bg-gray-600"></div>
            </div>
        </header>
    );
};

export default Header;
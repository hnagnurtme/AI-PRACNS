import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export type PageName = 'dashboard' | 'topology' | 'monitor' | 'comparison' | 'batch';

const Header: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();

    const getTabClassName = ( path: string ): string => {
        const baseStyle = "px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 flex items-center";
        const isActive = location.pathname === path || location.pathname === `/${ path }`;
        if ( isActive ) {
            return `${ baseStyle } bg-violet-600 text-white shadow-md`;
        }
        return `${ baseStyle } text-gray-300 hover:bg-gray-700/70 hover:text-white`;
    };

    return (
        <header className="w-full h-16 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 text-white flex items-center px-6 shadow-lg border-b border-gray-700 z-20">
            {/* Logo và Tiêu đề */ }
            <div className="flex items-center gap-3">
                <div className="relative">
                    <div className="absolute inset-0 bg-violet-500 blur-sm opacity-50 rounded-full"></div>
                    <svg
                        className="w-9 h-9 relative z-10 text-violet-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                </div>
                <div>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
                        SAGIN Routing
                    </h1>
                    <p className="text-[10px] text-gray-400 -mt-1">Space-Air-Ground Integrated Network</p>
                </div>
            </div>

            {/* Navigation tabs */ }
            <div className="ml-auto flex items-center gap-4">
                <nav className="flex items-center gap-1">
                    <button
                        className={ getTabClassName( '/dashboard' ) }
                        onClick={ () => navigate( '/dashboard' ) }
                    >
                        <svg className="w-4 h-4 inline-block mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                        </svg>
                        Dashboard
                    </button>
                    <button
                        className={ getTabClassName( '/topology' ) }
                        onClick={ () => navigate( '/topology' ) }
                    >
                        <svg className="w-4 h-4 inline-block mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        Topology
                    </button>
                    <button
                        className={ getTabClassName( '/monitor' ) }
                        onClick={ () => navigate( '/monitor' ) }
                    >
                        <svg className="w-4 h-4 inline-block mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Monitor
                    </button>
                    <button
                        className={ getTabClassName( '/comparison' ) }
                        onClick={ () => navigate( '/comparison' ) }
                    >
                        <svg className="w-4 h-4 inline-block mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Comparison
                    </button>
                    <button
                        className={ getTabClassName( '/batch' ) }
                        onClick={ () => navigate( '/batch' ) }
                    >
                        <svg className="w-4 h-4 inline-block mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                        </svg>
                        Batch
                    </button>
                </nav>

                {/* Status indicator */ }
                <div className="flex items-center gap-2 px-3 py-1.5 bg-violet-500/10 border border-violet-500/30 rounded-full">
                    <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-violet-400 font-medium">Online</span>
                </div>
            </div>
        </header>
    );
};

export default Header;
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export type PageName = 'dashboard' | 'topology' | 'monitor' | 'comparison' | 'batch';

const Header: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();

    const getTabClassName = ( path: string ): string => {
        const baseStyle = "px-4 py-2 text-sm font-medium rounded-xl transition-all duration-300 flex items-center gap-2";
        const isActive = location.pathname === path || location.pathname === `/${ path }`;
        if ( isActive ) {
            return `${ baseStyle } bg-gradient-to-r from-nebula-purple to-nebula-pink text-white shadow-nebula`;
        }
        return `${ baseStyle } text-star-silver hover:text-white hover:bg-white/10`;
    };

    return (
        <header className="w-full h-16 bg-cosmic-navy/80 backdrop-blur-lg text-white flex items-center px-6 border-b border-white/10 z-20 stars-bg">
            {/* Logo với Nebula Glow */ }
            <div className="flex items-center gap-4">
                <div className="relative group">
                    {/* Glow effect */ }
                    <div className="absolute inset-0 bg-nebula-purple blur-xl opacity-40 group-hover:opacity-60 transition-opacity rounded-full"></div>
                    <div className="relative w-10 h-10 bg-gradient-to-br from-nebula-purple to-nebula-pink rounded-xl flex items-center justify-center shadow-nebula">
                        <svg
                            className="w-6 h-6 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                </div>
                <div>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-nebula-purple via-nebula-pink to-nebula-cyan bg-clip-text text-transparent glow-text">
                        SAGIN Routing
                    </h1>
                    <p className="text-[10px] text-star-silver/70 -mt-0.5 tracking-wider uppercase">Space-Air-Ground Network</p>
                </div>
            </div>

            {/* Navigation - Cosmic Style */ }
            <div className="ml-auto flex items-center gap-6">
                <nav className="flex items-center gap-2 p-1 bg-white/5 rounded-2xl border border-white/10">
                    <button
                        className={ getTabClassName( '/dashboard' ) }
                        onClick={ () => navigate( '/dashboard' ) }
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                        </svg>
                        <span className="hidden md:inline">Dashboard</span>
                    </button>
                    <button
                        className={ getTabClassName( '/topology' ) }
                        onClick={ () => navigate( '/topology' ) }
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        <span className="hidden md:inline">Topology</span>
                    </button>
                    <button
                        className={ getTabClassName( '/monitor' ) }
                        onClick={ () => navigate( '/monitor' ) }
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        <span className="hidden md:inline">Monitor</span>
                    </button>
                    <button
                        className={ getTabClassName( '/comparison' ) }
                        onClick={ () => navigate( '/comparison' ) }
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                        </svg>
                        <span className="hidden md:inline">Compare</span>
                    </button>
                    <button
                        className={ getTabClassName( '/batch' ) }
                        onClick={ () => navigate( '/batch' ) }
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                        </svg>
                        <span className="hidden md:inline">Batch</span>
                    </button>
                </nav>

                {/* Status Indicator */ }
                <div className="flex items-center gap-2 px-4 py-2 glass-card">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(52,211,153,0.5)]"></div>
                    <span className="text-xs text-emerald-400 font-medium">Online</span>
                </div>
            </div>
        </header>
    );
};

export default Header;
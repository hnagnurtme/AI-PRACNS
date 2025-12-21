import React from 'react';

interface ColorLegendProps {
    className?: string;
}

export const ColorLegend: React.FC<ColorLegendProps> = ( { className = '' } ) => {
    return (
        <div className={ `bg-white/95 backdrop-blur-sm rounded-lg shadow-lg border border-gray-200 p-4 ${ className }` }>
            <h3 className="text-sm font-bold text-gray-800 mb-3">🎨 Color Legend</h3>

            {/* Routing Algorithms */ }
            <div className="mb-4">
                <p className="text-xs font-semibold text-gray-600 mb-2">Routing Paths:</p>
                <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-1 rounded-full bg-lime-400"></div>
                        <span className="text-xs text-gray-700">RL (Intelligent - Avoids Traps)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-1 rounded-full bg-orange-600"></div>
                        <span className="text-xs text-gray-700">Dijkstra (Shortest Path)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-1 rounded-full bg-teal-500"></div>
                        <span className="text-xs text-gray-700">Simple Routing</span>
                    </div>
                </div>
            </div>

            {/* Terminals */ }
            <div className="mb-4">
                <p className="text-xs font-semibold text-gray-600 mb-2">Terminals:</p>
                <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                        <span className="text-xs text-gray-700">Source Terminal</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-orange-600"></div>
                        <span className="text-xs text-gray-700">Destination Terminal</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                        <span className="text-xs text-gray-700">Transmitting</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                        <span className="text-xs text-gray-700">Connected</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                        <span className="text-xs text-gray-700">Idle</span>
                    </div>
                </div>
            </div>

            {/* Nodes */ }
            <div>
                <p className="text-xs font-semibold text-gray-600 mb-2">Network Nodes:</p>
                <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-cyan-400"></div>
                        <span className="text-xs text-gray-700">LEO Satellite</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                        <span className="text-xs text-gray-700">MEO Satellite</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                        <span className="text-xs text-gray-700">GEO Satellite</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                        <span className="text-xs text-gray-700">Ground Station</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

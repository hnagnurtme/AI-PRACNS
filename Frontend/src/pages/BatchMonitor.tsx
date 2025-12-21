import React, { useState, useEffect } from 'react';
import { useBatchPolling } from '../hooks/useBatchPolling';
import { BatchComparisonLog } from '../components/batchchart/BatchComparisonLog';
import { getBatchSuggestions, type BatchSuggestion } from '../services/batchService';

const BatchDashboard: React.FC = () => {
    // Lấy dữ liệu lô gói tin từ polling endpoint
    const { receivedBatches, connectionStatus } = useBatchPolling( true );

    // Lấy lô gói tin MỚI NHẤT
    const latestBatch = receivedBatches.length > 0 ? receivedBatches[ receivedBatches.length - 1 ] : null;

    // State for suggestions
    const [ suggestions, setSuggestions ] = useState<BatchSuggestion | null>( null );
    const [ _loadingSuggestions, setLoadingSuggestions ] = useState( false );


    // Load suggestions on mount
    useEffect( () => {
        const loadSuggestions = async () => {
            setLoadingSuggestions( true );
            try {
                const data = await getBatchSuggestions();
                setSuggestions( data );
            } catch ( err ) {
                console.error( 'Error loading suggestions:', err );
            } finally {
                setLoadingSuggestions( false );
            }
        };
        loadSuggestions();
    }, [] );



    if ( connectionStatus === 'DISCONNECTED' ) {
        return (
            <div className="p-10 text-center bg-cosmic-black min-h-screen">
                <div className="glass-card p-8 max-w-2xl mx-auto border-red-500/30">
                    <div className="flex items-center justify-center gap-4 mb-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-red-500 rounded-full opacity-20 animate-ping"></div>
                            <svg className="w-12 h-12 text-red-400 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3" />
                            </svg>
                        </div>
                        <h3 className="text-2xl font-bold text-red-400">Backend Disconnected</h3>
                    </div>
                    <p className="text-red-300 mb-2">Unable to connect to batch service</p>
                    <p className="text-xs text-red-400/70">Please ensure the backend server is running</p>
                </div>
            </div>
        );
    }

    if ( connectionStatus === 'CONNECTING' ) {
        return (
            <div className="p-10 text-center bg-cosmic-black min-h-screen">
                <div className="nebula-card p-8 max-w-2xl mx-auto">
                    <div className="flex items-center justify-center gap-4 mb-4">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-nebula-purple/30 border-t-nebula-purple"></div>
                        <h3 className="text-2xl font-bold bg-gradient-to-r from-nebula-purple to-nebula-pink bg-clip-text text-transparent">Connecting...</h3>
                    </div>
                    <p className="text-nebula-pink">Establishing connection to batch service</p>
                </div>
            </div>
        );
    }

    if ( !latestBatch ) {
        return (
            <div className="p-10 text-center bg-cosmic-black min-h-screen">
                <div className="glass-card p-6 max-w-2xl mx-auto">
                    <div className="flex items-center justify-center gap-3 mb-4">
                        <svg className="w-8 h-8 text-nebula-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <h3 className="text-lg font-semibold text-white">Waiting for Batch Data</h3>
                    </div>
                    <p className="text-sm text-emerald-400 mb-2">
                        ✅ Backend Connected (Polling)
                    </p>
                    <p className="text-xs text-star-silver/70">
                        Waiting for the first batch of packets from backend...
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 p-6 max-w-full mx-auto bg-cosmic-black min-h-screen cosmic-scrollbar">
            {/* Suggestions Section */ }
            { suggestions && (
                <>
                    {/* Test Case Suggestions - Single Row */ }
                    { suggestions.suggestedPairs && suggestions.suggestedPairs.length > 0 && (
                        <div className="glass-card p-6">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="bg-nebula-purple/20 p-2 rounded-lg">
                                    <span className="text-lg">🎯</span>
                                </div>
                                <h3 className="text-lg font-bold text-white">Test Case Suggestions</h3>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                                { suggestions.suggestedPairs.slice( 0, 4 ).map( ( pair, idx ) => (
                                    <div key={ idx } className="glass-card p-3">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className={ `px-2 py-1 rounded text-xs font-bold ${ pair.priority === 'high' ? 'bg-red-500/20 text-red-400' :
                                                pair.priority === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                                                    'bg-white/10 text-star-silver'
                                                }` }>
                                                { pair.type.replace( '_', ' ' ).toUpperCase() }
                                            </span>
                                            <span className="text-sm text-star-silver">{ pair.distance_km.toFixed( 0 ) } km</span>
                                        </div>
                                        <p className="text-sm font-medium text-white">
                                            { pair.sourceTerminalName } → { pair.destTerminalName }
                                        </p>
                                        <p className="text-xs text-star-silver/70 mt-1">{ pair.reason }</p>
                                    </div>
                                ) ) }
                            </div>
                        </div>
                    ) }

                    {/* Node Placement - Separate Full Width Row with 3 Columns */ }
                    { suggestions.nodePlacementRecommendations && suggestions.nodePlacementRecommendations.locations && suggestions.nodePlacementRecommendations.locations.length > 0 && (
                        <div className="glass-card p-6">
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-3">
                                    <div className="bg-teal-500/20 p-2 rounded-lg">
                                        <span className="text-lg">🗺️</span>
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-white">Recommended Node Placements</h3>
                                        <p className="text-sm text-star-silver/70">Top 3 locations for new nodes to improve coverage</p>
                                    </div>
                                </div>
                                <div className="flex gap-4">
                                    <div className="bg-teal-500/10 border border-teal-500/20 rounded-lg px-4 py-2 text-center">
                                        <p className="text-xs text-teal-400 font-medium">Current Coverage</p>
                                        <p className="text-xl font-bold text-teal-400">{ suggestions.nodePlacementRecommendations.currentCoverage.toFixed( 0 ) }%</p>
                                    </div>
                                    <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-2 text-center">
                                        <p className="text-xs text-amber-400 font-medium">Gaps Found</p>
                                        <p className="text-xl font-bold text-amber-400">{ suggestions.nodePlacementRecommendations.gapsIdentified }</p>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                { suggestions.nodePlacementRecommendations.locations.slice( 0, 3 ).map( ( location, idx ) => (
                                    <div key={ idx } className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                                        {/* Map Preview */ }
                                        <div className="h-40 bg-gradient-to-br from-teal-900/30 to-teal-800/20 relative">
                                            <iframe
                                                title={ `Map location ${ idx + 1 }` }
                                                src={ `https://www.openstreetmap.org/export/embed.html?bbox=${ location.longitude - 0.5 }%2C${ location.latitude - 0.5 }%2C${ location.longitude + 0.5 }%2C${ location.latitude + 0.5 }&layer=mapnik&marker=${ location.latitude }%2C${ location.longitude }` }
                                                className="w-full h-full border-0"
                                                loading="lazy"
                                            />
                                            <div className="absolute top-2 left-2 px-2 py-1 bg-teal-600 text-white text-xs font-bold rounded shadow">
                                                #{ location.rank } - { location.type }
                                            </div>
                                            <div className="absolute bottom-2 right-2 px-2 py-1 bg-cosmic-navy/90 text-star-silver text-xs font-medium rounded shadow">
                                                📍 { location.latitude.toFixed( 4 ) }°, { location.longitude.toFixed( 4 ) }°
                                            </div>
                                        </div>

                                        {/* Location Details */ }
                                        <div className="p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <span className="text-sm font-bold text-white">Priority Score</span>
                                                <span className={ `px-2 py-1 rounded text-xs font-bold ${ location.priorityScore >= 8 ? 'bg-red-500/20 text-red-400' :
                                                    location.priorityScore >= 5 ? 'bg-amber-500/20 text-amber-400' :
                                                        'bg-emerald-500/20 text-emerald-400'
                                                    }` }>
                                                    { location.priorityScore.toFixed( 1 ) }
                                                </span>
                                            </div>

                                            <div className="space-y-3">
                                                <div>
                                                    <p className="text-xs font-bold text-star-silver uppercase tracking-wide mb-1">📋 Reason</p>
                                                    <p className="text-sm text-star-silver/80">{ location.reason }</p>
                                                </div>

                                                <div>
                                                    <p className="text-xs font-bold text-teal-400 uppercase tracking-wide mb-1">✅ Expected Improvement</p>
                                                    <p className="text-sm text-teal-400/80">
                                                        { location.type === 'GROUND_STATION'
                                                            ? 'Reduces latency for nearby terminals, provides redundancy'
                                                            : location.type === 'LEO_SATELLITE'
                                                                ? 'Better coverage for remote areas, lower hop count'
                                                                : 'Improves relay capacity and signal strength' }
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ) ) }
                            </div>
                        </div>
                    ) }
                </>
            ) }


            {/* Results Section */ }
            { latestBatch ? (
                <BatchComparisonLog batch={ latestBatch } />
            ) : (
                <div className="bg-gradient-to-br from-white to-violet-50 rounded-2xl shadow-lg border-2 border-violet-200 p-12 text-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="bg-gradient-to-br from-violet-100 to-fuchsia-100 p-4 rounded-2xl">
                            <svg className="w-16 h-16 text-violet-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4\" />
                            </svg>
                        </div>
                        <div>
                            <p className="text-xl font-bold text-gray-700 mb-2">No Batch Data</p>
                            <p className="text-sm text-gray-500">Click "Generate" to create a new batch comparison</p>
                        </div>
                    </div>
                </div>
            ) }
        </div>
    );
};

export default BatchDashboard;
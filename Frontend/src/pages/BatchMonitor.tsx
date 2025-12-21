import React, { useState, useEffect } from 'react';
import { useBatchPolling } from '../hooks/useBatchPolling';
import { BatchComparisonLog } from '../components/batchchart/BatchComparisonLog';
import { generateBatch, getBatchSuggestions, type BatchSuggestion } from '../services/batchService';
import { getScenarios } from '../services/simulationService';
import type { SimulationScenario } from '../types/SimulationTypes';

const BatchDashboard: React.FC = () => {
    // Lấy dữ liệu lô gói tin từ polling endpoint
    const { receivedBatches, connectionStatus } = useBatchPolling( true );

    // Lấy lô gói tin MỚI NHẤT
    const latestBatch = receivedBatches.length > 0 ? receivedBatches[ receivedBatches.length - 1 ] : null;

    // State for test form
    const [ scenarios, setScenarios ] = useState<SimulationScenario[]>( [] );
    const [ selectedScenario, setSelectedScenario ] = useState<string>( 'NORMAL' );
    const [ pairCount, setPairCount ] = useState<number>( 10 );
    const [ loading, setLoading ] = useState( false );
    const [ error, setError ] = useState<string | null>( null );
    const [ suggestions, setSuggestions ] = useState<BatchSuggestion | null>( null );
    const [ _loadingSuggestions, setLoadingSuggestions ] = useState( false );

    // Load scenarios
    useEffect( () => {
        const loadScenarios = async () => {
            try {
                const data = await getScenarios();
                setScenarios( data );
                if ( data.length > 0 && !data.find( s => s.name === 'NORMAL' ) ) {
                    setSelectedScenario( data[ 0 ].name );
                }
            } catch ( err ) {
                console.error( 'Error loading scenarios:', err );
            }
        };
        loadScenarios();
    }, [] );

    // Load suggestions on mount
    useEffect( () => {
        const loadSuggestions = async () => {
            setLoadingSuggestions( true );
            try {
                const data = await getBatchSuggestions();
                setSuggestions( data );
                setPairCount( data.suggestedPairCount ); // Auto-set suggested count
            } catch ( err ) {
                console.error( 'Error loading suggestions:', err );
            } finally {
                setLoadingSuggestions( false );
            }
        };
        loadSuggestions();
    }, [] );

    const handleGenerate = async () => {
        setLoading( true );
        setError( null );
        try {
            await generateBatch( {
                pairCount,
                scenario: selectedScenario,
            } );
            // Batch will be picked up by polling
        } catch ( err ) {
            setError( err instanceof Error ? err.message : 'Failed to generate batch' );
        } finally {
            setLoading( false );
        }
    };

    if ( connectionStatus === 'DISCONNECTED' ) {
        return (
            <div className="p-10 text-center">
                <div className="bg-gradient-to-br from-red-50 to-pink-50 border-2 border-red-300 rounded-2xl p-8 max-w-2xl mx-auto shadow-xl">
                    <div className="flex items-center justify-center gap-4 mb-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-red-500 rounded-full opacity-20 animate-ping"></div>
                            <svg className="w-12 h-12 text-red-600 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3" />
                            </svg>
                        </div>
                        <h3 className="text-2xl font-bold bg-gradient-to-r from-red-600 to-pink-600 bg-clip-text text-transparent">Backend Disconnected</h3>
                    </div>
                    <p className="text-red-700 mb-2">Unable to connect to batch service</p>
                    <p className="text-xs text-red-500">Please ensure the backend server is running</p>
                </div>
            </div>
        );
    }

    if ( connectionStatus === 'CONNECTING' ) {
        return (
            <div className="p-10 text-center">
                <div className="bg-gradient-to-br from-violet-50 to-fuchsia-50 border-2 border-violet-300 rounded-2xl p-8 max-w-2xl mx-auto shadow-xl">
                    <div className="flex items-center justify-center gap-4 mb-4">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-violet-200 border-t-violet-600"></div>
                        <h3 className="text-2xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent">Connecting...</h3>
                    </div>
                    <p className="text-fuchsia-600">Establishing connection to batch service</p>
                </div>
            </div>
        );
    }

    if ( !latestBatch ) {
        return (
            <div className="p-10 text-center">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 max-w-2xl mx-auto">
                    <div className="flex items-center justify-center gap-3 mb-4">
                        <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <h3 className="text-lg font-semibold text-blue-800">Waiting for Batch Data</h3>
                    </div>
                    <p className="text-sm text-blue-700 mb-2">
                        ✅ Backend Connected (Polling)
                    </p>
                    <p className="text-xs text-blue-600">
                        Waiting for the first batch of packets from backend...
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 p-6 max-w-full mx-auto">
            {/* Suggestions Section */ }
            { suggestions && (
                <>
                    {/* Test Case Suggestions - Single Row */ }
                    { suggestions.suggestedPairs && suggestions.suggestedPairs.length > 0 && (
                        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="bg-violet-100 p-2 rounded-lg">
                                    <span className="text-lg">🎯</span>
                                </div>
                                <h3 className="text-lg font-bold text-gray-800">Test Case Suggestions</h3>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                                { suggestions.suggestedPairs.slice( 0, 4 ).map( ( pair, idx ) => (
                                    <div key={ idx } className="bg-gray-50 border border-gray-200 rounded-xl p-3">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className={ `px-2 py-1 rounded text-xs font-bold ${ pair.priority === 'high' ? 'bg-red-100 text-red-700' :
                                                pair.priority === 'medium' ? 'bg-amber-100 text-amber-700' :
                                                    'bg-gray-100 text-gray-700'
                                                }` }>
                                                { pair.type.replace( '_', ' ' ).toUpperCase() }
                                            </span>
                                            <span className="text-sm text-gray-600">{ pair.distance_km.toFixed( 0 ) } km</span>
                                        </div>
                                        <p className="text-sm font-medium text-gray-900">
                                            { pair.sourceTerminalName } → { pair.destTerminalName }
                                        </p>
                                        <p className="text-xs text-gray-500 mt-1">{ pair.reason }</p>
                                    </div>
                                ) ) }
                            </div>
                        </div>
                    ) }

                    {/* Node Placement - Separate Full Width Row with 3 Columns */ }
                    { suggestions.nodePlacementRecommendations && suggestions.nodePlacementRecommendations.locations && suggestions.nodePlacementRecommendations.locations.length > 0 && (
                        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-3">
                                    <div className="bg-teal-100 p-2 rounded-lg">
                                        <span className="text-lg">🗺️</span>
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-gray-800">Recommended Node Placements</h3>
                                        <p className="text-sm text-gray-500">Top 3 locations for new nodes to improve coverage</p>
                                    </div>
                                </div>
                                <div className="flex gap-4">
                                    <div className="bg-teal-50 border border-teal-200 rounded-lg px-4 py-2 text-center">
                                        <p className="text-xs text-teal-600 font-medium">Current Coverage</p>
                                        <p className="text-xl font-bold text-teal-700">{ suggestions.nodePlacementRecommendations.currentCoverage.toFixed( 0 ) }%</p>
                                    </div>
                                    <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 text-center">
                                        <p className="text-xs text-amber-600 font-medium">Gaps Found</p>
                                        <p className="text-xl font-bold text-amber-700">{ suggestions.nodePlacementRecommendations.gapsIdentified }</p>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                { suggestions.nodePlacementRecommendations.locations.slice( 0, 3 ).map( ( location, idx ) => (
                                    <div key={ idx } className="bg-gray-50 border border-gray-200 rounded-xl overflow-hidden">
                                        {/* Map Preview */ }
                                        <div className="h-40 bg-gradient-to-br from-teal-100 to-teal-50 relative">
                                            <iframe
                                                title={ `Map location ${ idx + 1 }` }
                                                src={ `https://www.openstreetmap.org/export/embed.html?bbox=${ location.longitude - 0.5 }%2C${ location.latitude - 0.5 }%2C${ location.longitude + 0.5 }%2C${ location.latitude + 0.5 }&layer=mapnik&marker=${ location.latitude }%2C${ location.longitude }` }
                                                className="w-full h-full border-0"
                                                loading="lazy"
                                            />
                                            <div className="absolute top-2 left-2 px-2 py-1 bg-teal-600 text-white text-xs font-bold rounded shadow">
                                                #{ location.rank } - { location.type }
                                            </div>
                                            <div className="absolute bottom-2 right-2 px-2 py-1 bg-white/90 text-gray-700 text-xs font-medium rounded shadow">
                                                📍 { location.latitude.toFixed( 4 ) }°, { location.longitude.toFixed( 4 ) }°
                                            </div>
                                        </div>

                                        {/* Location Details */ }
                                        <div className="p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <span className="text-sm font-bold text-gray-800">Priority Score</span>
                                                <span className={ `px-2 py-1 rounded text-xs font-bold ${ location.priorityScore >= 8 ? 'bg-red-100 text-red-700' :
                                                    location.priorityScore >= 5 ? 'bg-amber-100 text-amber-700' :
                                                        'bg-green-100 text-green-700'
                                                    }` }>
                                                    { location.priorityScore.toFixed( 1 ) }
                                                </span>
                                            </div>

                                            <div className="space-y-3">
                                                <div>
                                                    <p className="text-xs font-bold text-gray-600 uppercase tracking-wide mb-1">📋 Reason</p>
                                                    <p className="text-sm text-gray-700">{ location.reason }</p>
                                                </div>

                                                <div>
                                                    <p className="text-xs font-bold text-teal-600 uppercase tracking-wide mb-1">✅ Expected Improvement</p>
                                                    <p className="text-sm text-teal-700">
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
            {/* Generate Batch Form */ }
            <div className="bg-gradient-to-br from-white to-violet-50 rounded-2xl shadow-lg border-2 border-violet-200 p-6">
                <div className="flex items-center gap-3 mb-6">
                    <div className="bg-gradient-to-br from-violet-500 to-fuchsia-500 p-2 rounded-xl">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent uppercase tracking-wide">Generate Batch</h3>
                </div>

                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label htmlFor="scenario" className="block text-xs font-semibold text-fuchsia-600 uppercase tracking-wider mb-2">
                            Scenario
                        </label>
                        <select
                            id="scenario"
                            value={ selectedScenario }
                            onChange={ ( e ) => setSelectedScenario( e.target.value ) }
                            className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-white font-medium transition-all hover:border-violet-300"
                            disabled={ loading }
                        >
                            { scenarios.map( ( scenario ) => (
                                <option key={ scenario.name } value={ scenario.name }>
                                    { scenario.displayName }
                                </option>
                            ) ) }
                        </select>
                    </div>
                    <div>
                        <label htmlFor="pairCount" className="block text-xs font-semibold text-fuchsia-600 uppercase tracking-wider mb-2">
                            Pair Count
                        </label>
                        <input
                            type="number"
                            id="pairCount"
                            min="1"
                            max="50"
                            value={ pairCount }
                            onChange={ ( e ) => setPairCount( Number( e.target.value ) ) }
                            className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 font-medium transition-all hover:border-violet-300"
                            disabled={ loading }
                        />
                    </div>
                    <div className="flex items-end">
                        <button
                            onClick={ handleGenerate }
                            disabled={ loading }
                            className="w-full px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-bold rounded-xl hover:from-violet-700 hover:to-fuchsia-700 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg uppercase tracking-wide"
                        >
                            { loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                                    Generating...
                                </span>
                            ) : 'Generate' }
                        </button>
                    </div>
                </div>
                { error && (
                    <div className="mt-4 p-3 bg-red-50 border-2 border-red-200 rounded-xl">
                        <p className="text-sm text-red-700 font-medium">{ error }</p>
                    </div>
                ) }
            </div>

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
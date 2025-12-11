import React, { useState, useEffect } from 'react';
import { useBatchPolling } from '../hooks/useBatchPolling';
import { BatchComparisonLog } from '../components/batchchart/BatchComparisonLog';
import { generateBatch, getBatchSuggestions, type BatchSuggestion } from '../services/batchService';
import { getScenarios } from '../services/simulationService';
import type { SimulationScenario } from '../types/SimulationTypes';

const BatchDashboard: React.FC = () => {
    // Lấy dữ liệu lô gói tin từ polling endpoint
    const { receivedBatches, connectionStatus } = useBatchPolling(true);
    
    // Lấy lô gói tin MỚI NHẤT
    const latestBatch = receivedBatches.length > 0 ? receivedBatches[receivedBatches.length - 1] : null;

    // State for test form
    const [scenarios, setScenarios] = useState<SimulationScenario[]>([]);
    const [selectedScenario, setSelectedScenario] = useState<string>('NORMAL');
    const [pairCount, setPairCount] = useState<number>(10);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [suggestions, setSuggestions] = useState<BatchSuggestion | null>(null);
    const [_loadingSuggestions, setLoadingSuggestions] = useState(false);

    // Load scenarios
    useEffect(() => {
        const loadScenarios = async () => {
            try {
                const data = await getScenarios();
                setScenarios(data);
                if (data.length > 0 && !data.find(s => s.name === 'NORMAL')) {
                    setSelectedScenario(data[0].name);
                }
            } catch (err) {
                console.error('Error loading scenarios:', err);
            }
        };
        loadScenarios();
    }, []);

    // Load suggestions on mount
    useEffect(() => {
        const loadSuggestions = async () => {
            setLoadingSuggestions(true);
            try {
                const data = await getBatchSuggestions();
                setSuggestions(data);
                setPairCount(data.suggestedPairCount); // Auto-set suggested count
            } catch (err) {
                console.error('Error loading suggestions:', err);
            } finally {
                setLoadingSuggestions(false);
            }
        };
        loadSuggestions();
    }, []);

    const handleGenerate = async () => {
        setLoading(true);
        setError(null);
        try {
            await generateBatch({
                pairCount,
                scenario: selectedScenario,
            });
            // Batch will be picked up by polling
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to generate batch');
        } finally {
            setLoading(false);
        }
    };

    if (connectionStatus === 'DISCONNECTED') {
        return (
            <div className="p-10 text-center">
                <div className="bg-gradient-to-br from-red-50 to-pink-50 border-2 border-red-300 rounded-2xl p-8 max-w-2xl mx-auto shadow-xl">
                    <div className="flex items-center justify-center gap-4 mb-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-red-500 rounded-full opacity-20 animate-ping"></div>
                            <svg className="w-12 h-12 text-red-600 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3" />
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

    if (connectionStatus === 'CONNECTING') {
        return (
            <div className="p-10 text-center">
                <div className="bg-gradient-to-br from-violet-50 to-fuchsia-50 border-2 border-violet-300 rounded-2xl p-8 max-w-2xl mx-auto shadow-xl">
                    <div className="flex items-center justify-center gap-4 mb-4">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-violet-200 border-t-violet-600"></div>
                        <h3 className="text-2xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent">Connecting...</h3>
                    </div>
                    <p className="text-violet-700">Establishing connection to batch service</p>
                </div>
            </div>
        );
    }

    if (!latestBatch) {
        return (
            <div className="p-10 text-center">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 max-w-2xl mx-auto">
                    <div className="flex items-center justify-center gap-3 mb-4">
                        <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
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
            {/* Header with Connection Status */}
            <div className="bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-2xl shadow-2xl p-6 mb-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="bg-white/20 backdrop-blur-sm p-3 rounded-xl">
                            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                            </svg>
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-white tracking-tight">BATCH MONITOR</h1>
                            <p className="text-violet-100 text-sm mt-1">Real-time Packet Generation & Analysis</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3 bg-white/20 backdrop-blur-sm px-4 py-2 rounded-xl">
                        <div className="relative">
                            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                            <div className="absolute inset-0 bg-green-400 rounded-full animate-ping opacity-75"></div>
                        </div>
                        <span className="text-white font-semibold">CONNECTED</span>
                    </div>
                </div>
            </div>
            {/* Network Health Summary */}
            {suggestions && (
                <div className="bg-gradient-to-br from-white to-violet-50 rounded-2xl shadow-lg border-2 border-violet-200 p-6">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="bg-gradient-to-br from-violet-500 to-fuchsia-500 p-2 rounded-xl">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <h3 className="text-xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent uppercase tracking-wide">Network Health</h3>
                    </div>
                    
                    <div className="grid grid-cols-5 gap-3 mb-4">
                        <div className="bg-white/70 backdrop-blur-sm border-2 border-violet-200 rounded-xl p-3 text-center hover:scale-105 transition-transform">
                            <p className="text-xs text-violet-600 font-semibold uppercase tracking-wider mb-1">Nodes</p>
                            <p className="text-2xl font-bold text-gray-900">{suggestions.networkHealth.totalNodes}</p>
                        </div>
                        <div className="bg-white/70 backdrop-blur-sm border-2 border-violet-200 rounded-xl p-3 text-center hover:scale-105 transition-transform">
                            <p className="text-xs text-violet-600 font-semibold uppercase tracking-wider mb-1">Terminals</p>
                            <p className="text-2xl font-bold text-gray-900">{suggestions.networkHealth.totalTerminals}</p>
                        </div>
                        <div className="bg-white/70 backdrop-blur-sm border-2 border-red-200 rounded-xl p-3 text-center hover:scale-105 transition-transform">
                            <p className="text-xs text-red-600 font-semibold uppercase tracking-wider mb-1">Overloaded</p>
                            <p className="text-2xl font-bold text-red-600">{suggestions.networkHealth.overloadedNodes}</p>
                        </div>
                        <div className="bg-white/70 backdrop-blur-sm border-2 border-green-200 rounded-xl p-3 text-center hover:scale-105 transition-transform">
                            <p className="text-xs text-green-600 font-semibold uppercase tracking-wider mb-1">Capacity</p>
                            <p className={`text-2xl font-bold ${
                                suggestions.networkHealth.availableCapacity > 70 ? 'text-green-600' :
                                suggestions.networkHealth.availableCapacity > 50 ? 'text-yellow-600' :
                                'text-red-600'
                            }`}>
                                {suggestions.networkHealth.availableCapacity.toFixed(0)}%
                            </p>
                        </div>
                        <div className="bg-white/70 backdrop-blur-sm border-2 border-fuchsia-200 rounded-xl p-3 text-center hover:scale-105 transition-transform">
                            <p className="text-xs text-fuchsia-600 font-semibold uppercase tracking-wider mb-1">Suggested</p>
                            <p className="text-2xl font-bold text-fuchsia-600">{suggestions.suggestedPairCount}</p>
                            <p className="text-xs text-gray-500">pairs</p>
                        </div>
                    </div>

                    {/* Top 3 Recommendations Only */}
                    {suggestions.recommendations && suggestions.recommendations.length > 0 && (
                        <div className="space-y-2">
                            {suggestions.recommendations.slice(0, 3).map((rec, idx) => (
                                <div key={idx} className={`p-3 rounded-xl border-2 backdrop-blur-sm ${
                                    rec.type === 'error' ? 'bg-red-50/80 border-red-300' :
                                    rec.type === 'warning' ? 'bg-yellow-50/80 border-yellow-300' :
                                    rec.type === 'success' ? 'bg-green-50/80 border-green-300' :
                                    'bg-violet-50/80 border-violet-300'
                                }`}>
                                    <p className={`text-sm font-semibold ${
                                        rec.type === 'error' ? 'text-red-800' :
                                        rec.type === 'warning' ? 'text-yellow-800' :
                                        rec.type === 'success' ? 'text-green-800' :
                                        'text-violet-800'
                                    }`}>
                                        {rec.message}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Suggested Pairs - Compact View */}
                    {suggestions.suggestedPairs && suggestions.suggestedPairs.length > 0 && (
                        <div className="mt-4">
                            <h4 className="text-sm font-bold text-violet-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span>🎯</span> Suggested Test Pairs
                            </h4>
                            <div className="grid grid-cols-2 gap-3">
                                {suggestions.suggestedPairs.slice(0, 4).map((pair, idx) => (
                                    <div key={idx} className="bg-white/70 backdrop-blur-sm border-2 border-violet-200 rounded-xl p-3 hover:scale-105 transition-transform">
                                        <div className="flex items-start justify-between mb-2">
                                            <span className={`px-2 py-1 rounded-lg text-xs font-bold ${
                                                pair.priority === 'high' ? 'bg-red-500 text-white' :
                                                pair.priority === 'medium' ? 'bg-yellow-500 text-white' :
                                                'bg-blue-500 text-white'
                                            }`}>
                                                {pair.type.replace('_', ' ').toUpperCase()}
                                            </span>
                                        </div>
                                        <p className="text-sm font-bold text-gray-900 mb-1">
                                            {pair.sourceTerminalName} → {pair.destTerminalName}
                                        </p>
                                        <p className="text-xs text-violet-600 font-semibold mb-1">{pair.distance_km.toFixed(0)} km</p>
                                        <p className="text-xs text-gray-600 italic">{pair.reason}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Link Quality Prediction - Compact */}
                    {suggestions.linkQualityPrediction && (
                        <div className="mt-4 bg-white/70 backdrop-blur-sm border-2 border-indigo-200 rounded-xl p-4">
                            <h4 className="text-sm font-bold text-indigo-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span>📡</span> Link Quality (Next 6h)
                            </h4>
                            <div className="text-xs mb-3">
                                <p className="text-gray-700 font-medium">
                                    {suggestions.linkQualityPrediction.pairInfo.source} → {suggestions.linkQualityPrediction.pairInfo.destination}
                                    <span className="ml-2 text-indigo-600">({suggestions.linkQualityPrediction.pairInfo.distance_km} km)</span>
                                </p>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <div className="bg-green-100 border-2 border-green-300 rounded-xl p-2 text-center">
                                    <p className="text-xs text-green-700 font-semibold mb-1">Best</p>
                                    <p className="text-lg font-bold text-green-800">
                                        {new Date(suggestions.linkQualityPrediction.bestTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                    <p className="text-xs text-green-600">{suggestions.linkQualityPrediction.bestQuality}</p>
                                </div>
                                <div className="bg-blue-100 border-2 border-blue-300 rounded-xl p-2 text-center">
                                    <p className="text-xs text-blue-700 font-semibold mb-1">Avg</p>
                                    <p className="text-lg font-bold text-blue-800">
                                        {suggestions.linkQualityPrediction.averageQuality}
                                    </p>
                                    <p className="text-xs text-blue-600">Quality</p>
                                </div>
                                <div className="bg-red-100 border-2 border-red-300 rounded-xl p-2 text-center">
                                    <p className="text-xs text-red-700 font-semibold mb-1">Worst</p>
                                    <p className="text-lg font-bold text-red-800">
                                        {new Date(suggestions.linkQualityPrediction.worstTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                    <p className="text-xs text-red-600">{suggestions.linkQualityPrediction.worstQuality}</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Node Placement - Compact Top 3 */}
                    {suggestions.nodePlacementRecommendations && suggestions.nodePlacementRecommendations.locations && suggestions.nodePlacementRecommendations.locations.length > 0 && (
                        <div className="mt-4 bg-white/70 backdrop-blur-sm border-2 border-purple-200 rounded-xl p-4">
                            <h4 className="text-sm font-bold text-purple-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span>🗺️</span> Node Placement (Top 3)
                            </h4>
                            <div className="grid grid-cols-2 gap-2 mb-3">
                                <div className="bg-purple-100 border-2 border-purple-300 rounded-lg p-2 text-center">
                                    <p className="text-xs text-purple-700 font-semibold">Coverage</p>
                                    <p className="text-xl font-bold text-purple-800">
                                        {suggestions.nodePlacementRecommendations.currentCoverage.toFixed(0)}%
                                    </p>
                                </div>
                                <div className="bg-orange-100 border-2 border-orange-300 rounded-lg p-2 text-center">
                                    <p className="text-xs text-orange-700 font-semibold">Gaps</p>
                                    <p className="text-xl font-bold text-orange-800">
                                        {suggestions.nodePlacementRecommendations.gapsIdentified}
                                    </p>
                                </div>
                            </div>
                            <div className="space-y-2">
                                {suggestions.nodePlacementRecommendations.locations.slice(0, 3).map((location, idx) => (
                                    <div key={idx} className="bg-gradient-to-r from-purple-50 to-indigo-50 border-2 border-purple-200 rounded-lg p-2">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="px-2 py-0.5 rounded-lg text-xs font-bold bg-purple-600 text-white">
                                                #{location.rank}
                                            </span>
                                            <span className="text-sm font-bold text-gray-900">{location.type}</span>
                                            <span className="text-xs text-gray-500">Score: {location.priorityScore.toFixed(1)}</span>
                                        </div>
                                        <p className="text-xs text-gray-600">
                                            📍 {location.latitude.toFixed(2)}°, {location.longitude.toFixed(2)}°
                                        </p>
                                        <p className="text-xs text-purple-700 italic mt-1">{location.reason}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Overload Analysis - Compact Top 5 */}
                    {suggestions.overloadAnalysis && suggestions.overloadAnalysis.overloadedNodes && suggestions.overloadAnalysis.overloadedNodes.length > 0 && (
                        <div className="mt-4 bg-white/70 backdrop-blur-sm border-2 border-red-200 rounded-xl p-4">
                            <h4 className="text-sm font-bold text-red-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span>⚠️</span> Overloaded Nodes ({suggestions.overloadAnalysis.overloadedNodes.length})
                            </h4>
                            <div className="space-y-2">
                                {suggestions.overloadAnalysis.overloadedNodes.slice(0, 5).map((node, idx) => (
                                    <div key={idx} className="bg-red-50 border-2 border-red-200 rounded-lg p-2">
                                        <div className="flex items-center justify-between mb-1">
                                            <p className="text-sm font-bold text-gray-900">{node.nodeName}</p>
                                            <span className={`px-2 py-1 rounded-lg text-xs font-bold ${
                                                node.overloadScore >= 0.8 ? 'bg-red-600 text-white' :
                                                node.overloadScore >= 0.5 ? 'bg-orange-500 text-white' :
                                                'bg-yellow-500 text-white'
                                            }`}>
                                                {(node.overloadScore * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            <div className="bg-white rounded px-2 py-1">
                                                <span className="text-gray-600">Util: </span>
                                                <span className="font-bold text-red-700">{node.utilization.toFixed(0)}%</span>
                                            </div>
                                            <div className="bg-white rounded px-2 py-1">
                                                <span className="text-gray-600">Loss: </span>
                                                <span className="font-bold text-red-700">{node.packetLoss.toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Generate Batch Form */}
            <div className="bg-gradient-to-br from-white to-fuchsia-50 rounded-2xl shadow-lg border-2 border-violet-200 p-6">
                <div className="flex items-center gap-3 mb-6">
                    <div className="bg-gradient-to-br from-violet-500 to-fuchsia-500 p-2 rounded-xl">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent uppercase tracking-wide">Generate Batch</h3>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label htmlFor="scenario" className="block text-xs font-semibold text-violet-700 uppercase tracking-wider mb-2">
                            Scenario
                        </label>
                        <select
                            id="scenario"
                            value={selectedScenario}
                            onChange={(e) => setSelectedScenario(e.target.value)}
                            className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-white font-medium transition-all hover:border-violet-300"
                            disabled={loading}
                        >
                            {scenarios.map((scenario) => (
                                <option key={scenario.name} value={scenario.name}>
                                    {scenario.displayName}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label htmlFor="pairCount" className="block text-xs font-semibold text-violet-700 uppercase tracking-wider mb-2">
                            Pair Count
                        </label>
                        <input
                            type="number"
                            id="pairCount"
                            min="1"
                            max="50"
                            value={pairCount}
                            onChange={(e) => setPairCount(Number(e.target.value))}
                            className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 font-medium transition-all hover:border-violet-300"
                            disabled={loading}
                        />
                    </div>
                    <div className="flex items-end">
                        <button
                            onClick={handleGenerate}
                            disabled={loading}
                            className="w-full px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-bold rounded-xl hover:from-violet-700 hover:to-fuchsia-700 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg uppercase tracking-wide"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                                    Generating...
                                </span>
                            ) : 'Generate'}
                        </button>
                    </div>
                </div>
                {error && (
                    <div className="mt-4 p-3 bg-red-50 border-2 border-red-200 rounded-xl">
                        <p className="text-sm text-red-700 font-medium">{error}</p>
                    </div>
                )}
            </div>

            {/* Results Section */}
            {latestBatch ? (
                <BatchComparisonLog batch={latestBatch} />
            ) : (
                <div className="bg-gradient-to-br from-white to-violet-50 rounded-2xl shadow-lg border-2 border-violet-200 p-12 text-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="bg-gradient-to-br from-violet-100 to-fuchsia-100 p-4 rounded-2xl">
                            <svg className="w-16 h-16 text-violet-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4\" />
                            </svg>
                        </div>
                        <div>
                            <p className="text-xl font-bold text-gray-700 mb-2">No Batch Data</p>
                            <p className="text-sm text-gray-500">Click "Generate" to create a new batch comparison</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default BatchDashboard;
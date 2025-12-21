import React, { useState, useEffect } from "react";
import { compareAlgorithms } from "../services/routingService";
import { getUserTerminals } from "../services/userTerminalService";
import { PacketRouteGraph } from "../components/chart/PacketRouteGraph";
import { CombinedHopMetricsChart } from "../components/chart/CombinedHopMetricsChart";

import type { AlgorithmComparison, CompareAlgorithmsRequest } from "../types/RoutingTypes";
import type { UserTerminal, QoSRequirements } from "../types/UserTerminalTypes";
import type { ComparisonData } from "../types/ComparisonTypes";

export const ComparisonDashboard: React.FC = () => {
    // State for terminals
    const [ terminals, setTerminals ] = useState<UserTerminal[]>( [] );
    const [ loadingTerminals, setLoadingTerminals ] = useState( false );

    // Form state
    const [ sourceTerminalId, setSourceTerminalId ] = useState<string>( "" );
    const [ destinationTerminalId, setDestinationTerminalId ] = useState<string>( "" );
    const [ algorithm1, setAlgorithm1 ] = useState<'simple' | 'dijkstra' | 'rl'>( 'dijkstra' );
    const [ algorithm2, setAlgorithm2 ] = useState<'simple' | 'dijkstra' | 'rl'>( 'rl' );
    const [ serviceQos, setServiceQos ] = useState<QoSRequirements>( {
        maxLatencyMs: 100,
        minBandwidthMbps: 1,
        maxLossRate: 0.01,
        priority: 5
    } );

    // Results state
    const [ comparisonResult, setComparisonResult ] = useState<AlgorithmComparison | null>( null );
    const [ loading, setLoading ] = useState( false );
    const [ error, setError ] = useState<string | null>( null );
    const [ successMessage, setSuccessMessage ] = useState<string | null>( null );

    // Load terminals on mount
    useEffect( () => {
        const loadData = async () => {
            setLoadingTerminals( true );
            try {
                const terminalsData = await getUserTerminals();
                setTerminals( terminalsData );
            } catch ( err ) {
                console.error( 'Failed to load terminals:', err );
                setError( err instanceof Error ? err.message : 'Failed to load terminals' );
            } finally {
                setLoadingTerminals( false );
            }
        };
        loadData();
    }, [] );

    // Handle form submission
    const handleCompare = async ( e: React.FormEvent ) => {
        e.preventDefault();

        if ( !sourceTerminalId || !destinationTerminalId ) {
            setError( 'Please select both source and destination terminals' );
            return;
        }

        if ( sourceTerminalId === destinationTerminalId ) {
            setError( 'Source and destination terminals must be different' );
            return;
        }

        if ( algorithm1 === algorithm2 ) {
            setError( 'Please select two different algorithms' );
            return;
        }

        setLoading( true );
        setError( null );
        setSuccessMessage( null );

        try {
            const request: CompareAlgorithmsRequest = {
                sourceTerminalId,
                destinationTerminalId,
                algorithm1,
                algorithm2,
                serviceQos,
                scenario: 'NORMAL'
            };

            const result = await compareAlgorithms( request );
            setComparisonResult( result );

            // Show success message
            setSuccessMessage( '✅ Comparison completed and saved! View results in History page.' );
            console.log( '✅ Comparison saved to database and can be viewed in History page' );
        } catch ( err ) {
            setError( err instanceof Error ? err.message : 'Failed to compare algorithms' );
            setComparisonResult( null );
        } finally {
            setLoading( false );
        }
    };

    // Convert comparison result to ComparisonData format for charts
    const convertToComparisonData = ( result: AlgorithmComparison | null ): ComparisonData | null => {
        if ( !result ) return null;

        // Helper function to convert path to packet format
        const pathToPacket = ( path: typeof result.algorithm1.path, algorithmName: string, isRL: boolean ) => {
            const algorithmLabel = algorithmName === 'dijkstra' ? 'Dijkstra' : algorithmName === 'rl' ? 'ReinforcementLearning' : 'Simple';
            return {
                packetId: `PKT-${ algorithmName }`,
                sourceUserId: result.sourceTerminalId,
                destinationUserId: result.destinationTerminalId,
                stationSource: result.sourceTerminalId,
                stationDest: result.destinationTerminalId,
                type: 'comparison',
                timeSentFromSourceMs: Date.now(),
                payloadDataBase64: '',
                payloadSizeByte: 1024,
                serviceQoS: {
                    serviceType: 'TEXT_MESSAGE' as const,
                    defaultPriority: serviceQos.priority,
                    maxLatencyMs: serviceQos.maxLatencyMs,
                    maxJitterMs: 10,
                    minBandwidthMbps: serviceQos.minBandwidthMbps,
                    maxLossRate: serviceQos.maxLossRate
                },
                currentHoldingNodeId: path.path[ path.path.length - 1 ]?.id || '',
                nextHopNodeId: '',
                pathHistory: path.path.map( p => p.id ),
                hopRecords: path.path.slice( 1 ).map( ( segment, idx ) => {
                    const prevSegment = path.path[ idx ];
                    const segmentDistance = prevSegment && segment.position ?
                        Math.sqrt(
                            Math.pow( ( segment.position.latitude - prevSegment.position.latitude ) * 111000, 2 ) +
                            Math.pow( ( segment.position.longitude - prevSegment.position.longitude ) * 111000 * Math.cos( prevSegment.position.latitude * Math.PI / 180 ), 2 ) +
                            Math.pow( ( segment.position.altitude - prevSegment.position.altitude ) / 1000, 2 )
                        ) : 0;

                    // Get node resource data from comparison result
                    const nodeResource = result.nodeResources?.[ prevSegment?.id ];
                    const bandwidthUtil = nodeResource?.resourceUtilization ? nodeResource.resourceUtilization / 100 : Math.random() * 0.6 + 0.2; // Mock: 20-80%
                    const queueSize = nodeResource?.currentPacketCount || Math.floor( Math.random() * 50 + 10 ); // Mock: 10-60 packets

                    return {
                        fromNodeId: prevSegment?.id || '',
                        toNodeId: segment.id,
                        latencyMs: path.hops > 0 ? path.estimatedLatency / path.hops : 0,
                        timestampMs: Date.now() + idx * 10,
                        distanceKm: segmentDistance,
                        fromNodePosition: prevSegment?.position || null,
                        toNodePosition: segment.position,
                        fromNodeBufferState: { queueSize, bandwidthUtilization: bandwidthUtil },
                        routingDecisionInfo: { algorithm: algorithmLabel as "Dijkstra" | "ReinforcementLearning" }
                    };
                } ),
                accumulatedDelayMs: path.estimatedLatency,
                priorityLevel: serviceQos.priority,
                maxAcceptableLatencyMs: serviceQos.maxLatencyMs,
                maxAcceptableLossRate: serviceQos.maxLossRate,
                dropped: false,
                analysisData: {
                    avgLatency: path.estimatedLatency,
                    avgDistanceKm: path.totalDistance,
                    routeSuccessRate: 1,
                    totalDistanceKm: path.totalDistance,
                    totalLatencyMs: path.estimatedLatency
                },
                isUseRL: isRL,
                TTL: 100
            };
        };

        // Map algorithm1 to dijkstraPacket and algorithm2 to rlPacket
        // (charts expect this structure, regardless of actual algorithm names)
        return {
            dijkstraPacket: pathToPacket( result.algorithm1.path, result.algorithm1.name, result.algorithm1.name === 'rl' ),
            rlPacket: pathToPacket( result.algorithm2.path, result.algorithm2.name, result.algorithm2.name === 'rl' )
        };
    };

    const comparisonData = convertToComparisonData( comparisonResult );

    return (
        <div className="p-6 space-y-6 bg-gradient-to-br from-slate-50 via-violet-50 to-fuchsia-50 min-h-screen">


            {/* Form Section */ }
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border-2 border-violet-200">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-3 bg-gradient-to-br from-violet-600 to-fuchsia-600 rounded-xl shadow-lg">
                        <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <div>
                        <h2 className="text-3xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent">Algorithm Comparison Monitor</h2>
                        <p className="text-sm text-violet-600 font-medium">Compare routing algorithms with different network scenarios</p>
                    </div>
                </div>

                <form onSubmit={ handleCompare } className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Source Terminal */ }
                        <div>
                            <label htmlFor="sourceTerminal" className="block text-sm font-semibold text-fuchsia-600 mb-2 uppercase tracking-wide">
                                📡 Source Terminal
                            </label>
                            <select
                                id="sourceTerminal"
                                value={ sourceTerminalId }
                                onChange={ ( e ) => setSourceTerminalId( e.target.value ) }
                                className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-white"
                                disabled={ loading || loadingTerminals }
                                required
                            >
                                <option value="">Select source terminal...</option>
                                { terminals.map( ( terminal ) => (
                                    <option key={ terminal.terminalId } value={ terminal.terminalId }>
                                        { terminal.terminalName } ({ terminal.terminalId })
                                    </option>
                                ) ) }
                            </select>
                        </div>

                        {/* Destination Terminal */ }
                        <div>
                            <label htmlFor="destinationTerminal" className="block text-sm font-semibold text-fuchsia-600 mb-2 uppercase tracking-wide">
                                🎯 Destination Terminal
                            </label>
                            <select
                                id="destinationTerminal"
                                value={ destinationTerminalId }
                                onChange={ ( e ) => setDestinationTerminalId( e.target.value ) }
                                className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-white"
                                disabled={ loading || loadingTerminals }
                                required
                            >
                                <option value="">Select destination terminal...</option>
                                { terminals.map( ( terminal ) => (
                                    <option key={ terminal.terminalId } value={ terminal.terminalId }>
                                        { terminal.terminalName } ({ terminal.terminalId })
                                    </option>
                                ) ) }
                            </select>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

                        {/* Algorithm 1 */ }
                        <div>
                            <label htmlFor="algorithm1" className="block text-sm font-semibold text-fuchsia-600 mb-2 uppercase tracking-wide">
                                <span className="flex items-center gap-1">
                                    <span>🔷</span>
                                    <span>Algorithm 1</span>
                                </span>
                            </label>
                            <select
                                id="algorithm1"
                                value={ algorithm1 }
                                onChange={ ( e ) => setAlgorithm1( e.target.value as 'simple' | 'dijkstra' | 'rl' ) }
                                className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-white"
                                disabled={ loading }
                                required
                            >
                                <option value="simple">Simple</option>
                                <option value="dijkstra">Dijkstra</option>
                                <option value="rl">Reinforcement Learning</option>
                            </select>
                        </div>

                        {/* Algorithm 2 */ }
                        <div>
                            <label htmlFor="algorithm2" className="block text-sm font-semibold text-fuchsia-600 mb-2 uppercase tracking-wide">
                                <span className="flex items-center gap-1">
                                    <span>🔶</span>
                                    <span>Algorithm 2</span>
                                </span>
                            </label>
                            <select
                                id="algorithm2"
                                value={ algorithm2 }
                                onChange={ ( e ) => setAlgorithm2( e.target.value as 'simple' | 'dijkstra' | 'rl' ) }
                                className="block w-full px-4 py-3 border-2 border-violet-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-white"
                                disabled={ loading }
                                required
                            >
                                <option value="simple">Simple</option>
                                <option value="dijkstra">Dijkstra</option>
                                <option value="rl">Reinforcement Learning</option>
                            </select>
                        </div>
                    </div>

                    {/* Service QoS */ }
                    <div className="bg-gradient-to-r from-violet-50 to-fuchsia-50 p-4 rounded-xl border border-violet-200">
                        <h3 className="text-sm font-bold text-violet-800 mb-3 uppercase tracking-wide">⚙️ QoS Requirements</h3>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div>
                                <label htmlFor="maxLatency" className="block text-xs font-semibold text-fuchsia-600 mb-2 uppercase">
                                    Max Latency
                                </label>
                                <div className="relative">
                                    <input
                                        type="number"
                                        id="maxLatency"
                                        value={ serviceQos.maxLatencyMs }
                                        onChange={ ( e ) => setServiceQos( { ...serviceQos, maxLatencyMs: Number( e.target.value ) } ) }
                                        className="block w-full px-4 py-2 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                                        disabled={ loading }
                                        min="1"
                                    />
                                    <span className="absolute right-3 top-2 text-xs text-gray-500 font-medium">ms</span>
                                </div>
                            </div>
                            <div>
                                <label htmlFor="minBandwidth" className="block text-xs font-semibold text-fuchsia-600 mb-2 uppercase">
                                    Min Bandwidth
                                </label>
                                <div className="relative">
                                    <input
                                        type="number"
                                        id="minBandwidth"
                                        value={ serviceQos.minBandwidthMbps }
                                        onChange={ ( e ) => setServiceQos( { ...serviceQos, minBandwidthMbps: Number( e.target.value ) } ) }
                                        className="block w-full px-4 py-2 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                                        disabled={ loading }
                                        min="0.1"
                                        step="0.1"
                                    />
                                    <span className="absolute right-3 top-2 text-xs text-gray-500 font-medium">Mbps</span>
                                </div>
                            </div>
                            <div>
                                <label htmlFor="maxLossRate" className="block text-xs font-semibold text-fuchsia-600 mb-2 uppercase">
                                    Max Loss Rate
                                </label>
                                <div className="relative">
                                    <input
                                        type="number"
                                        id="maxLossRate"
                                        value={ serviceQos.maxLossRate }
                                        onChange={ ( e ) => setServiceQos( { ...serviceQos, maxLossRate: Number( e.target.value ) } ) }
                                        className="block w-full px-4 py-2 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                                        disabled={ loading }
                                        min="0"
                                        max="1"
                                        step="0.01"
                                    />
                                    <span className="absolute right-3 top-2 text-xs text-gray-500 font-medium">rate</span>
                                </div>
                            </div>
                            <div>
                                <label htmlFor="priority" className="block text-xs font-semibold text-fuchsia-600 mb-2 uppercase">
                                    Priority Level
                                </label>
                                <input
                                    type="number"
                                    id="priority"
                                    value={ serviceQos.priority }
                                    onChange={ ( e ) => setServiceQos( { ...serviceQos, priority: Number( e.target.value ) } ) }
                                    className="block w-full px-4 py-2 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                                    disabled={ loading }
                                    min="1"
                                    max="10"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Submit Button */ }
                    <div className="flex justify-end pt-6 border-t-2 border-violet-200 mt-6">
                        <button
                            type="submit"
                            disabled={ loading || loadingTerminals || !sourceTerminalId || !destinationTerminalId }
                            className="px-10 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white rounded-xl hover:from-violet-700 hover:to-fuchsia-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-xl hover:shadow-2xl font-bold flex items-center gap-3 text-lg"
                        >
                            { loading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Comparing...
                                </>
                            ) : (
                                <>
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                    </svg>
                                    Compare Algorithms
                                </>
                            ) }
                        </button>
                    </div>
                </form>

                { error && (
                    <div className="mt-4 p-3 bg-red-50 rounded-md border border-red-200">
                        <p className="text-sm text-red-800">{ error }</p>
                    </div>
                ) }

                { successMessage && (
                    <div className="mt-4 p-3 bg-green-50 rounded-md border border-green-200">
                        <p className="text-sm text-green-800">{ successMessage }</p>
                    </div>
                ) }
            </div>

            {/* Comparison Results */ }
            { comparisonResult && (
                <>
                    {/* Comparison Summary */ }
                    <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl p-6 border-2 border-violet-200">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-3">
                                <h3 className="text-2xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent">Comparison Results</h3>
                                { comparisonResult.scenario && comparisonResult.scenario !== 'NORMAL' && (
                                    <span className="inline-block px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-semibold">
                                        { comparisonResult.scenario.replace( /_/g, ' ' ) }
                                    </span>
                                ) }
                            </div>
                            <div className="text-xs text-gray-500">
                                { new Date( comparisonResult.timestamp ).toLocaleString() }
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                            {/* Algorithm 1 Results - Dynamic color based on algorithm type */ }
                            <div className={ `border-2 rounded-xl p-4 shadow-md ${ comparisonResult.algorithm1.name === 'rl'
                                ? 'border-teal-300 bg-gradient-to-br from-teal-50 to-white'
                                : 'border-violet-300 bg-gradient-to-br from-violet-50 to-white'
                                }` }>
                                <h4 className={ `font-bold text-lg mb-3 flex items-center gap-2 ${ comparisonResult.algorithm1.name === 'rl' ? 'text-teal-800' : 'text-violet-800'
                                    }` }>
                                    <span className={ `w-3 h-3 rounded-full shadow-sm ${ comparisonResult.algorithm1.name === 'rl' ? 'bg-teal-500' : 'bg-violet-500'
                                        }` }></span>
                                    { comparisonResult.algorithm1.name === 'rl' ? 'Reinforcement Learning' :
                                        comparisonResult.algorithm1.name === 'dijkstra' ? 'Dijkstra' :
                                            comparisonResult.algorithm1.name.charAt( 0 ).toUpperCase() + comparisonResult.algorithm1.name.slice( 1 ) }
                                </h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm1.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>Distance:</span>
                                        <span className="font-bold text-gray-900">{ comparisonResult.algorithm1.path.totalDistance.toFixed( 0 ) } km</span>
                                    </div>
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm1.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>Latency:</span>
                                        <span className="font-bold text-gray-900">{ comparisonResult.algorithm1.path.estimatedLatency.toFixed( 0 ) } ms</span>
                                    </div>
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm1.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>Hops:</span>
                                        <span className="font-bold text-gray-900">{ comparisonResult.algorithm1.path.hops }</span>
                                    </div>
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm1.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>QoS Met:</span>
                                        <span className={ `font-bold px-3 py-1 rounded-lg ${ comparisonResult.algorithm1.qosMet ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700' }` }>
                                            { comparisonResult.algorithm1.qosMet ? '✓ Yes' : '✗ No' }
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Algorithm 2 Results - Dynamic color based on algorithm type */ }
                            <div className={ `border-2 rounded-xl p-4 shadow-md ${ comparisonResult.algorithm2.name === 'rl'
                                ? 'border-teal-300 bg-gradient-to-br from-teal-50 to-white'
                                : 'border-violet-300 bg-gradient-to-br from-violet-50 to-white'
                                }` }>
                                <h4 className={ `font-bold text-lg mb-3 flex items-center gap-2 ${ comparisonResult.algorithm2.name === 'rl' ? 'text-teal-800' : 'text-violet-800'
                                    }` }>
                                    <span className={ `w-3 h-3 rounded-full shadow-sm ${ comparisonResult.algorithm2.name === 'rl' ? 'bg-teal-500' : 'bg-violet-500'
                                        }` }></span>
                                    { comparisonResult.algorithm2.name === 'rl' ? 'Reinforcement Learning' :
                                        comparisonResult.algorithm2.name === 'dijkstra' ? 'Dijkstra' :
                                            comparisonResult.algorithm2.name.charAt( 0 ).toUpperCase() + comparisonResult.algorithm2.name.slice( 1 ) }
                                </h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm2.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>Distance:</span>
                                        <span className="font-bold text-gray-900">{ comparisonResult.algorithm2.path.totalDistance.toFixed( 0 ) } km</span>
                                    </div>
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm2.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>Latency:</span>
                                        <span className="font-bold text-gray-900">{ comparisonResult.algorithm2.path.estimatedLatency.toFixed( 0 ) } ms</span>
                                    </div>
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm2.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>Hops:</span>
                                        <span className="font-bold text-gray-900">{ comparisonResult.algorithm2.path.hops }</span>
                                    </div>
                                    <div className="flex justify-between bg-white/50 px-2 py-1.5 rounded-lg">
                                        <span className={ `font-semibold ${ comparisonResult.algorithm2.name === 'rl' ? 'text-teal-700' : 'text-fuchsia-600' }` }>QoS Met:</span>
                                        <span className={ `font-bold px-3 py-1 rounded-lg ${ comparisonResult.algorithm2.qosMet ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700' }` }>
                                            { comparisonResult.algorithm2.qosMet ? '✓ Yes' : '✗ No' }
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Comparison Metrics */ }
                        <div className="border-t-2 border-violet-200 pt-4 mt-4">
                            <h4 className="font-bold text-base text-violet-800 mb-3 flex items-center gap-2">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                Performance Comparison
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                                <div className="bg-gradient-to-br from-violet-50 to-fuchsia-50 p-3 rounded-xl border border-violet-200 shadow-sm">
                                    <div className="text-fuchsia-600 font-semibold mb-1 uppercase text-xs">🏆 Best Distance</div>
                                    <div className="font-bold text-xl text-violet-900">
                                        { comparisonResult.comparison.bestDistance === 'rl' ? 'Reinforcement Learning' :
                                            comparisonResult.comparison.bestDistance === 'dijkstra' ? 'Dijkstra' :
                                                comparisonResult.comparison.bestDistance.charAt( 0 ).toUpperCase() + comparisonResult.comparison.bestDistance.slice( 1 ) }
                                    </div>
                                    <div className="text-xs text-violet-600 mt-1 font-medium">
                                        Δ { Math.abs( comparisonResult.comparison.distanceDifference ).toFixed( 0 ) } km
                                    </div>
                                </div>
                                <div className="bg-gradient-to-br from-green-50 to-green-100 p-3 rounded-xl border border-green-200 shadow-sm">
                                    <div className="text-green-700 font-semibold mb-1 uppercase text-xs">⚡ Best Latency</div>
                                    <div className="font-bold text-xl text-green-900">
                                        { comparisonResult.comparison.bestLatency === 'rl' ? 'Reinforcement Learning' :
                                            comparisonResult.comparison.bestLatency === 'dijkstra' ? 'Dijkstra' :
                                                comparisonResult.comparison.bestLatency.charAt( 0 ).toUpperCase() + comparisonResult.comparison.bestLatency.slice( 1 ) }
                                    </div>
                                    <div className="text-xs text-green-600 mt-1 font-medium">
                                        Δ { Math.abs( comparisonResult.comparison.latencyDifference ).toFixed( 0 ) } ms
                                    </div>
                                </div>
                                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-3 rounded-xl border border-purple-200 shadow-sm">
                                    <div className="text-purple-700 font-semibold mb-1 uppercase text-xs">🔗 Best Hops</div>
                                    <div className="font-bold text-xl text-purple-900">
                                        { comparisonResult.comparison.bestHops === 'rl' ? 'Reinforcement Learning' :
                                            comparisonResult.comparison.bestHops === 'dijkstra' ? 'Dijkstra' :
                                                comparisonResult.comparison.bestHops.charAt( 0 ).toUpperCase() + comparisonResult.comparison.bestHops.slice( 1 ) }
                                    </div>
                                    <div className="text-xs text-purple-600 mt-1 font-medium">
                                        Δ { Math.abs( comparisonResult.comparison.hopsDifference ) } hops
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* QoS Warnings */ }
                        { comparisonResult.qosWarnings.length > 0 && (
                            <div className="mt-4 p-3 bg-yellow-50 rounded-md border border-yellow-200">
                                <h5 className="font-semibold text-yellow-800 mb-2">QoS Warnings</h5>
                                <ul className="list-disc list-inside text-sm text-yellow-700">
                                    { comparisonResult.qosWarnings.map( ( warning, idx ) => (
                                        <li key={ idx }>{ warning }</li>
                                    ) ) }
                                </ul>
                            </div>
                        ) }
                    </div>

                    {/* Charts */ }
                    { comparisonData && (
                        <>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                { comparisonData.dijkstraPacket && (
                                    <PacketRouteGraph data={ comparisonData } />
                                ) }
                            </div>

                            <CombinedHopMetricsChart
                                data={ comparisonData }
                                scenario={ comparisonResult.scenario || 'NORMAL' }
                            />
                        </>
                    ) }
                </>
            ) }

            { !comparisonResult && !loading && (
                <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl p-12 border-2 border-violet-200">
                    <div className="text-center mb-8">
                        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-violet-100 to-fuchsia-100 rounded-full mb-4">
                            <svg className="w-10 h-10 text-violet-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <h3 className="text-2xl font-bold text-gray-800 mb-2">Ready to Compare Algorithms</h3>
                        <p className="text-violet-600 font-medium">Select terminals and algorithms above to analyze routing performance</p>
                    </div>

                    {/* Mock Data Preview */ }
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 opacity-50">
                        <div className="border-2 border-violet-200 rounded-2xl p-6 bg-gradient-to-br from-violet-50 to-white">
                            <h4 className="font-bold text-xl text-violet-800 mb-4 flex items-center gap-2">
                                <span className="w-4 h-4 rounded-full bg-violet-500"></span>
                                Algorithm A (Example)
                            </h4>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-fuchsia-600 font-semibold">Distance:</span>
                                    <span className="font-bold text-gray-900">12,450 km</span>
                                </div>
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-fuchsia-600 font-semibold">Latency:</span>
                                    <span className="font-bold text-gray-900">125 ms</span>
                                </div>
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-fuchsia-600 font-semibold">Hops:</span>
                                    <span className="font-bold text-gray-900">8</span>
                                </div>
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-fuchsia-600 font-semibold">QoS Met:</span>
                                    <span className="font-bold px-3 py-1 rounded-lg bg-green-100 text-green-700">✓ Yes</span>
                                </div>
                            </div>
                        </div>

                        <div className="border-2 border-pink-200 rounded-2xl p-6 bg-gradient-to-br from-pink-50 to-white">
                            <h4 className="font-bold text-xl text-pink-800 mb-4 flex items-center gap-2">
                                <span className="w-4 h-4 rounded-full bg-pink-500"></span>
                                Algorithm B (Example)
                            </h4>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-pink-700 font-semibold">Distance:</span>
                                    <span className="font-bold text-gray-900">11,820 km</span>
                                </div>
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-pink-700 font-semibold">Latency:</span>
                                    <span className="font-bold text-gray-900">98 ms</span>
                                </div>
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-pink-700 font-semibold">Hops:</span>
                                    <span className="font-bold text-gray-900">6</span>
                                </div>
                                <div className="flex justify-between bg-white/50 p-2 rounded-lg">
                                    <span className="text-pink-700 font-semibold">QoS Met:</span>
                                    <span className="font-bold px-3 py-1 rounded-lg bg-green-100 text-green-700">✓ Yes</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 text-center text-sm text-gray-500">
                        <p>💡 The comparison will show detailed metrics, charts, and node resource analysis</p>
                    </div>
                </div>
            ) }
        </div>
    );
};

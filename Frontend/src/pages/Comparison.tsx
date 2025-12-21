import React, { useState, useEffect, useMemo } from "react";
import { compareAlgorithms } from "../services/routingService";
import { getUserTerminals } from "../services/userTerminalService";
import type { AlgorithmComparison, CompareAlgorithmsRequest, NodeResourceInfo } from "../types/RoutingTypes";
import type { UserTerminal, QoSRequirements } from "../types/UserTerminalTypes";

const Comparison: React.FC = () => {
    // State for terminals
    const [ terminals, setTerminals ] = useState<UserTerminal[]>( [] );
    const [ loadingTerminals, setLoadingTerminals ] = useState( false );

    // State for scenarios (selectedScenario used internally)
    const [ selectedScenario ] = useState<string>( 'NORMAL' );

    // Form state
    const [ sourceTerminalId, setSourceTerminalId ] = useState<string>( "" );
    const [ destinationTerminalId, setDestinationTerminalId ] = useState<string>( "" );
    const [ algorithm1, setAlgorithm1 ] = useState<'dijkstra' | 'rl'>( 'dijkstra' );
    const [ algorithm2, setAlgorithm2 ] = useState<'dijkstra' | 'rl'>( 'rl' );
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

    // Filter and sort state
    const [ filterStatus, setFilterStatus ] = useState<'all' | 'critical' | 'warning' | 'normal'>( 'all' );
    const [ sortBy, setSortBy ] = useState<'name' | 'utilization' | 'packetLoss' | 'queue'>( 'name' );
    const [ showOnlyAffected, setShowOnlyAffected ] = useState( true );

    // Load terminals and scenarios on mount
    useEffect( () => {
        const loadData = async () => {
            setLoadingTerminals( true );
            try {
                const terminalsData = await getUserTerminals();
                setTerminals( terminalsData );
            } catch ( err ) {
                console.error( 'Failed to load data:', err );
                setError( err instanceof Error ? err.message : 'Failed to load data' );
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

        try {
            const request: CompareAlgorithmsRequest = {
                sourceTerminalId,
                destinationTerminalId,
                algorithm1,
                algorithm2,
                serviceQos,
                scenario: selectedScenario
            };

            const result = await compareAlgorithms( request );
            setComparisonResult( result );
        } catch ( err ) {
            setError( err instanceof Error ? err.message : 'Failed to compare algorithms' );
            setComparisonResult( null );
        } finally {
            setLoading( false );
        }
    };

    // Get warning status for a node
    const getWarningStatus = ( node: NodeResourceInfo ) => {
        const warnings: string[] = [];
        let status: 'critical' | 'warning' | 'normal' = 'normal';

        // Check latency
        if ( node.nodeProcessingDelayMs > 500 ) {
            warnings.push( 'High Latency' );
            status = 'critical';
        } else if ( node.nodeProcessingDelayMs > 200 ) {
            warnings.push( 'Moderate Latency' );
            if ( status === 'normal' ) status = 'warning';
        }

        // Check packet loss
        if ( node.packetLossRate > 0.1 ) {
            warnings.push( 'High Packet Loss' );
            status = 'critical';
        } else if ( node.packetLossRate > 0.05 ) {
            warnings.push( 'Moderate Packet Loss' );
            if ( status === 'normal' ) status = 'warning';
        }

        // Check queue
        const queueRatio = node.packetBufferCapacity > 0
            ? ( node.currentPacketCount / node.packetBufferCapacity ) * 100
            : 0;
        if ( queueRatio > 90 ) {
            warnings.push( 'Queue Nearly Full' );
            status = 'critical';
        } else if ( queueRatio > 70 ) {
            warnings.push( 'High Queue' );
            if ( status === 'normal' ) status = 'warning';
        }

        // Check utilization
        if ( node.resourceUtilization > 90 ) {
            warnings.push( 'High Utilization' );
            status = 'critical';
        } else if ( node.resourceUtilization > 70 ) {
            warnings.push( 'Moderate Utilization' );
            if ( status === 'normal' ) status = 'warning';
        }

        return { status, warnings, queueRatio };
    };

    // Process nodes for display
    const processedNodes = useMemo( () => {
        if ( !comparisonResult?.nodeResources ) return [];

        const path1NodeIds = comparisonResult.algorithm1.path.path
            .filter( seg => seg.type === 'node' )
            .map( seg => seg.id );
        const path2NodeIds = comparisonResult.algorithm2.path.path
            .filter( seg => seg.type === 'node' )
            .map( seg => seg.id );
        const allNodeIds = Array.from( new Set( [ ...path1NodeIds, ...path2NodeIds ] ) );

        let nodes = allNodeIds
            .map( nodeId => comparisonResult.nodeResources?.[ nodeId ] )
            .filter( ( node ): node is NodeResourceInfo => node !== undefined )
            .map( node => {
                const { status, warnings, queueRatio } = getWarningStatus( node );
                const inPath1 = path1NodeIds.includes( node.nodeId );
                const inPath2 = path2NodeIds.includes( node.nodeId );

                return {
                    ...node,
                    status,
                    warnings,
                    queueRatio,
                    inPath1,
                    inPath2,
                    isAffected: inPath1 || inPath2
                };
            } );

        // Filter by affected nodes
        if ( showOnlyAffected ) {
            nodes = nodes.filter( n => n.isAffected );
        }

        // Filter by status
        if ( filterStatus !== 'all' ) {
            nodes = nodes.filter( n => n.status === filterStatus );
        }

        // Sort
        nodes.sort( ( a, b ) => {
            switch ( sortBy ) {
                case 'utilization':
                    return b.resourceUtilization - a.resourceUtilization;
                case 'packetLoss':
                    return b.packetLossRate - a.packetLossRate;
                case 'queue':
                    return b.queueRatio - a.queueRatio;
                default:
                    return a.nodeName.localeCompare( b.nodeName );
            }
        } );

        return nodes;
    }, [ comparisonResult, filterStatus, sortBy, showOnlyAffected ] );

    // Statistics
    const statistics = useMemo( () => {
        if ( !comparisonResult?.nodeResources ) {
            return { total: 0, critical: 0, warning: 0, normal: 0, overloaded: 0, highPacketLoss: 0 };
        }

        const path1NodeIds = comparisonResult.algorithm1.path.path
            .filter( seg => seg.type === 'node' )
            .map( seg => seg.id );
        const path2NodeIds = comparisonResult.algorithm2.path.path
            .filter( seg => seg.type === 'node' )
            .map( seg => seg.id );
        const allNodeIds = Array.from( new Set( [ ...path1NodeIds, ...path2NodeIds ] ) );

        const nodes = allNodeIds
            .map( nodeId => comparisonResult.nodeResources?.[ nodeId ] )
            .filter( ( node ): node is NodeResourceInfo => node !== undefined );

        let critical = 0, warning = 0, normal = 0, overloaded = 0, highPacketLoss = 0;

        nodes.forEach( node => {
            const { status } = getWarningStatus( node );
            if ( status === 'critical' ) critical++;
            else if ( status === 'warning' ) warning++;
            else normal++;

            if ( node.resourceUtilization > 90 ) overloaded++;
            if ( node.packetLossRate > 0.1 ) highPacketLoss++;
        } );

        return {
            total: nodes.length,
            critical,
            warning,
            normal,
            overloaded,
            highPacketLoss
        };
    }, [ comparisonResult ] );

    return (
        <div className="p-6 space-y-6 bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen">
            {/* Header */ }
            <div className="bg-white rounded-xl shadow-xl p-6 border border-gray-200">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-gradient-to-br from-violet-600 to-fuchsia-600 rounded-lg">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-800">Node Impact Analysis</h1>
                        <p className="text-sm text-gray-500">Analyze affected nodes and resource utilization</p>
                    </div>
                </div>
            </div>

            {/* Form Section */ }
            <div className="bg-gradient-to-br from-white to-violet-50 rounded-xl shadow-xl p-6 border-2 border-violet-200">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-gradient-to-br from-violet-600 to-fuchsia-600 rounded-lg">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                        </svg>
                    </div>
                    <h2 className="text-lg font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent uppercase tracking-wide">
                        Configuration
                    </h2>
                </div>

                <form onSubmit={ handleCompare } className="space-y-4">
                    {/* Terminals Selection */ }
                    <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-violet-200">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-bold text-fuchsia-600 mb-2 uppercase tracking-wide flex items-center gap-1">
                                    <span>📡</span>
                                    <span>Source Terminal</span>
                                </label>
                                <select
                                    value={ sourceTerminalId }
                                    onChange={ ( e ) => setSourceTerminalId( e.target.value ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-medium transition-all"
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

                            <div>
                                <label className="block text-xs font-bold text-fuchsia-600 mb-2 uppercase tracking-wide flex items-center gap-1">
                                    <span>🎯</span>
                                    <span>Destination Terminal</span>
                                </label>
                                <select
                                    value={ destinationTerminalId }
                                    onChange={ ( e ) => setDestinationTerminalId( e.target.value ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-medium transition-all"
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
                    </div>

                    {/* Algorithms Selection */ }
                    <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-violet-200">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-bold text-violet-600 mb-2 uppercase tracking-wide flex items-center gap-1">
                                    <span>🔴</span>
                                    <span>Algorithm 1 (Dijkstra)</span>
                                </label>
                                <select
                                    value={ algorithm1 }
                                    onChange={ ( e ) => setAlgorithm1( e.target.value as 'dijkstra' | 'rl' ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-medium transition-all"
                                    disabled={ loading }
                                    required
                                >
                                    <option value="dijkstra">Dijkstra</option>
                                    <option value="rl">Reinforcement Learning</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-xs font-bold text-teal-600 mb-2 uppercase tracking-wide flex items-center gap-1">
                                    <span>�</span>
                                    <span>Algorithm 2 (RL)</span>
                                </label>
                                <select
                                    value={ algorithm2 }
                                    onChange={ ( e ) => setAlgorithm2( e.target.value as 'dijkstra' | 'rl' ) }
                                    className="block w-full px-3 py-2.5 border-2 border-teal-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent bg-white text-gray-800 font-medium transition-all"
                                    disabled={ loading }
                                    required
                                >
                                    <option value="dijkstra">Dijkstra</option>
                                    <option value="rl">Reinforcement Learning</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Service QoS */ }
                    <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-violet-200">
                        <div className="flex items-center gap-2 mb-3">
                            <span className="text-lg">⚙️</span>
                            <h3 className="text-xs font-bold text-fuchsia-600 uppercase tracking-wide">Quality of Service Parameters</h3>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-600 mb-2 flex items-center gap-1">
                                    <span>⏱️</span>
                                    <span>Max Latency (ms)</span>
                                </label>
                                <input
                                    type="number"
                                    value={ serviceQos.maxLatencyMs }
                                    onChange={ ( e ) => setServiceQos( { ...serviceQos, maxLatencyMs: Number( e.target.value ) } ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-semibold transition-all"
                                    disabled={ loading }
                                    min="1"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-600 mb-2 flex items-center gap-1">
                                    <span>📶</span>
                                    <span>Min Bandwidth (Mbps)</span>
                                </label>
                                <input
                                    type="number"
                                    value={ serviceQos.minBandwidthMbps }
                                    onChange={ ( e ) => setServiceQos( { ...serviceQos, minBandwidthMbps: Number( e.target.value ) } ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-semibold transition-all"
                                    disabled={ loading }
                                    min="0.1"
                                    step="0.1"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-600 mb-2 flex items-center gap-1">
                                    <span>📉</span>
                                    <span>Max Loss Rate</span>
                                </label>
                                <input
                                    type="number"
                                    value={ serviceQos.maxLossRate }
                                    onChange={ ( e ) => setServiceQos( { ...serviceQos, maxLossRate: Number( e.target.value ) } ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-semibold transition-all"
                                    disabled={ loading }
                                    min="0"
                                    max="1"
                                    step="0.01"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-600 mb-2 flex items-center gap-1">
                                    <span>⭐</span>
                                    <span>Priority</span>
                                </label>
                                <input
                                    type="number"
                                    value={ serviceQos.priority }
                                    onChange={ ( e ) => setServiceQos( { ...serviceQos, priority: Number( e.target.value ) } ) }
                                    className="block w-full px-3 py-2.5 border-2 border-violet-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent bg-white text-gray-800 font-semibold transition-all"
                                    disabled={ loading }
                                    min="1"
                                    max="10"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Submit Button */ }
                    <div className="flex justify-end">
                        <button
                            type="submit"
                            disabled={ loading || loadingTerminals || !sourceTerminalId || !destinationTerminalId }
                            className="group relative px-8 py-3.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white rounded-xl hover:from-violet-700 hover:to-fuchsia-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-2xl hover:scale-105 font-bold uppercase tracking-wide text-sm"
                        >
                            <span className="flex items-center gap-2">
                                { loading ? (
                                    <>
                                        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        <span>Analyzing...</span>
                                    </>
                                ) : (
                                    <>
                                        <svg className="w-5 h-5 group-hover:rotate-12 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                                        </svg>
                                        <span>Analyze Node Impact</span>
                                    </>
                                ) }
                            </span>
                        </button>
                    </div>
                </form>

                { error && (
                    <div className="mt-4 p-3 bg-red-50 rounded-md border border-red-200">
                        <p className="text-sm text-red-800">{ error }</p>
                    </div>
                ) }
            </div>

            {/* Results */ }
            { comparisonResult && (
                <>
                    {/* Statistics Cards - Simplified colors */ }
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                        <div className="bg-gray-50 rounded-lg shadow-md p-3 border border-gray-300">
                            <div className="text-xs text-gray-600 mb-1 font-medium uppercase tracking-wide">Total Nodes</div>
                            <div className="text-xl font-bold text-gray-800">{ statistics.total }</div>
                        </div>
                        <div className="bg-red-50 rounded-lg shadow-md p-3 border border-red-300">
                            <div className="text-xs text-red-700 mb-1 font-medium uppercase tracking-wide">Critical</div>
                            <div className="text-xl font-bold text-red-700">{ statistics.critical }</div>
                        </div>
                        <div className="bg-amber-50 rounded-lg shadow-md p-3 border border-amber-300">
                            <div className="text-xs text-amber-700 mb-1 font-medium uppercase tracking-wide">Warning</div>
                            <div className="text-xl font-bold text-amber-700">{ statistics.warning }</div>
                        </div>
                        <div className="bg-gray-50 rounded-lg shadow-md p-3 border border-gray-300">
                            <div className="text-xs text-red-600 mb-1 font-medium uppercase tracking-wide">Overloaded</div>
                            <div className="text-xl font-bold text-red-600">{ statistics.overloaded }</div>
                        </div>
                        <div className="bg-gray-50 rounded-lg shadow-md p-3 border border-gray-300">
                            <div className="text-xs text-red-600 mb-1 font-medium uppercase tracking-wide">High Loss</div>
                            <div className="text-xl font-bold text-red-600">{ statistics.highPacketLoss }</div>
                        </div>
                    </div>

                    {/* Filters and Controls */ }
                    <div className="bg-white rounded-lg shadow-md p-4 border border-gray-200">
                        <div className="flex flex-wrap items-center gap-3">
                            <div className="flex items-center gap-2">
                                <label className="text-sm font-medium text-gray-700">Filter Status:</label>
                                <select
                                    value={ filterStatus }
                                    onChange={ ( e ) => setFilterStatus( e.target.value as any ) }
                                    className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-violet-500 focus:border-blue-500"
                                >
                                    <option value="all">All</option>
                                    <option value="critical">Critical</option>
                                    <option value="warning">Warning</option>
                                    <option value="normal">Normal</option>
                                </select>
                            </div>

                            <div className="flex items-center gap-2">
                                <label className="text-sm font-medium text-gray-700">Sort By:</label>
                                <select
                                    value={ sortBy }
                                    onChange={ ( e ) => setSortBy( e.target.value as any ) }
                                    className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-violet-500 focus:border-blue-500"
                                >
                                    <option value="name">Name</option>
                                    <option value="utilization">Utilization</option>
                                    <option value="packetLoss">Packet Loss</option>
                                    <option value="queue">Queue</option>
                                </select>
                            </div>

                            <div className="flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    id="showOnlyAffected"
                                    checked={ showOnlyAffected }
                                    onChange={ ( e ) => setShowOnlyAffected( e.target.checked ) }
                                    className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                                />
                                <label htmlFor="showOnlyAffected" className="text-sm font-medium text-gray-700">
                                    Show Only Affected Nodes
                                </label>
                            </div>
                        </div>
                    </div>

                    {/* Detailed Node Table - Two Column Comparison */ }
                    <div className="bg-white rounded-xl shadow-xl border border-gray-200 p-6">
                        <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                            <span>🔍</span>
                            <span>Affected Nodes Analysis</span>
                        </h3>

                        { processedNodes.length === 0 ? (
                            <div className="text-center py-12 text-gray-500">
                                No nodes found matching the current filters
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 gap-6">
                                {/* LEFT COLUMN: Dijkstra Path Nodes */ }
                                <div className="space-y-3">
                                    <div className="bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-lg p-3">
                                        <h4 className="text-white font-bold text-sm uppercase tracking-wide flex items-center gap-2">
                                            <span>🔵</span>
                                            <span>{ comparisonResult.algorithm1.name === 'dijkstra' ? 'Dijkstra' : comparisonResult.algorithm1.name.toUpperCase() } - Shortest Path</span>
                                        </h4>
                                        <p className="text-violet-100 text-xs mt-1">Optimizes for distance only, may ignore resource constraints</p>
                                    </div>

                                    <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                                        { processedNodes.filter( n => n.inPath1 ).length === 0 ? (
                                            <div className="text-center py-6 text-gray-500 text-sm">
                                                No nodes in this path
                                            </div>
                                        ) : (
                                            processedNodes.filter( n => n.inPath1 ).map( ( node ) => {
                                                const isOverloaded = node.resourceUtilization > 90;
                                                const hasHighPacketLoss = node.packetLossRate > 0.1;
                                                const isQueueFull = node.queueRatio > 90;

                                                return (
                                                    <div
                                                        key={ node.nodeId }
                                                        className={ `p-3 rounded-lg border-2 transition-all hover:shadow-md ${ node.status === 'critical' ? 'bg-red-50 border-red-400' :
                                                            node.status === 'warning' ? 'bg-yellow-50 border-yellow-400' :
                                                                'bg-white border-violet-200'
                                                            }` }
                                                    >
                                                        <div className="flex items-start justify-between mb-2">
                                                            <div>
                                                                <div className="font-bold text-sm text-gray-900">{ node.nodeName }</div>
                                                                <div className="text-xs text-gray-500">{ node.nodeType }</div>
                                                            </div>
                                                            <span className={ `px-2 py-0.5 rounded text-xs font-bold ${ node.status === 'critical' ? 'bg-red-200 text-red-800' :
                                                                node.status === 'warning' ? 'bg-yellow-200 text-yellow-800' :
                                                                    'bg-green-100 text-green-700'
                                                                }` }>
                                                                { node.status === 'critical' ? '🔴 CRITICAL' :
                                                                    node.status === 'warning' ? '⚠️ WARNING' : '✅ OK' }
                                                            </span>
                                                        </div>

                                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                                            <div>
                                                                <div className="text-gray-500 font-medium">CPU/Resource</div>
                                                                <div className={ `font-bold ${ isOverloaded ? 'text-red-600' :
                                                                    node.resourceUtilization > 70 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { node.resourceUtilization.toFixed( 0 ) }%
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="text-gray-500 font-medium">Packet Loss</div>
                                                                <div className={ `font-bold ${ hasHighPacketLoss ? 'text-red-600' :
                                                                    node.packetLossRate > 0.05 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { ( node.packetLossRate * 100 ).toFixed( 1 ) }%
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="text-gray-500 font-medium">Queue</div>
                                                                <div className={ `font-bold ${ isQueueFull ? 'text-red-600' :
                                                                    node.queueRatio > 70 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { node.currentPacketCount }/{ node.packetBufferCapacity }
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="text-gray-500 font-medium">Latency</div>
                                                                <div className={ `font-bold ${ node.nodeProcessingDelayMs > 500 ? 'text-red-600' :
                                                                    node.nodeProcessingDelayMs > 200 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { node.nodeProcessingDelayMs.toFixed( 0 ) } ms
                                                                </div>
                                                            </div>
                                                        </div>

                                                        { node.warnings.length > 0 && (
                                                            <div className="mt-2 pt-2 border-t border-gray-200">
                                                                <div className="flex flex-wrap gap-1">
                                                                    { node.warnings.map( ( warning, idx ) => (
                                                                        <span
                                                                            key={ idx }
                                                                            className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-700 font-medium"
                                                                        >
                                                                            { warning }
                                                                        </span>
                                                                    ) ) }
                                                                </div>
                                                            </div>
                                                        ) }
                                                    </div>
                                                );
                                            } )
                                        ) }
                                    </div>
                                </div>

                                {/* RIGHT COLUMN: RL Path Nodes */ }
                                <div className="space-y-3">
                                    <div className="bg-gradient-to-r from-teal-600 to-teal-700 rounded-lg p-3">
                                        <h4 className="text-white font-bold text-sm uppercase tracking-wide flex items-center gap-2">
                                            <span>🟠</span>
                                            <span>{ comparisonResult.algorithm2.name === 'rl' ? 'Reinforcement Learning' : comparisonResult.algorithm2.name.toUpperCase() } - Balanced Path</span>
                                        </h4>
                                        <p className="text-teal-100 text-xs mt-1">Balances distance and resource utilization for optimal performance</p>
                                    </div>

                                    <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                                        { processedNodes.filter( n => n.inPath2 ).length === 0 ? (
                                            <div className="text-center py-6 text-gray-500 text-sm">
                                                No nodes in this path
                                            </div>
                                        ) : (
                                            processedNodes.filter( n => n.inPath2 ).map( ( node ) => {
                                                const isOverloaded = node.resourceUtilization > 90;
                                                const hasHighPacketLoss = node.packetLossRate > 0.1;
                                                const isQueueFull = node.queueRatio > 90;

                                                return (
                                                    <div
                                                        key={ node.nodeId }
                                                        className={ `p-3 rounded-lg border-2 transition-all hover:shadow-md ${ node.status === 'critical' ? 'bg-red-50 border-red-400' :
                                                            node.status === 'warning' ? 'bg-yellow-50 border-yellow-400' :
                                                                'bg-white border-teal-200'
                                                            }` }
                                                    >
                                                        <div className="flex items-start justify-between mb-2">
                                                            <div>
                                                                <div className="font-bold text-sm text-gray-900">{ node.nodeName }</div>
                                                                <div className="text-xs text-gray-500">{ node.nodeType }</div>
                                                            </div>
                                                            <span className={ `px-2 py-0.5 rounded text-xs font-bold ${ node.status === 'critical' ? 'bg-red-200 text-red-800' :
                                                                node.status === 'warning' ? 'bg-yellow-200 text-yellow-800' :
                                                                    'bg-green-100 text-green-700'
                                                                }` }>
                                                                { node.status === 'critical' ? '🔴 CRITICAL' :
                                                                    node.status === 'warning' ? '⚠️ WARNING' : '✅ OK' }
                                                            </span>
                                                        </div>

                                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                                            <div>
                                                                <div className="text-gray-500 font-medium">CPU/Resource</div>
                                                                <div className={ `font-bold ${ isOverloaded ? 'text-red-600' :
                                                                    node.resourceUtilization > 70 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { node.resourceUtilization.toFixed( 0 ) }%
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="text-gray-500 font-medium">Packet Loss</div>
                                                                <div className={ `font-bold ${ hasHighPacketLoss ? 'text-red-600' :
                                                                    node.packetLossRate > 0.05 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { ( node.packetLossRate * 100 ).toFixed( 1 ) }%
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="text-gray-500 font-medium">Queue</div>
                                                                <div className={ `font-bold ${ isQueueFull ? 'text-red-600' :
                                                                    node.queueRatio > 70 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { node.currentPacketCount }/{ node.packetBufferCapacity }
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="text-gray-500 font-medium">Latency</div>
                                                                <div className={ `font-bold ${ node.nodeProcessingDelayMs > 500 ? 'text-red-600' :
                                                                    node.nodeProcessingDelayMs > 200 ? 'text-yellow-600' :
                                                                        'text-green-600'
                                                                    }` }>
                                                                    { node.nodeProcessingDelayMs.toFixed( 0 ) } ms
                                                                </div>
                                                            </div>
                                                        </div>

                                                        { node.warnings.length > 0 && (
                                                            <div className="mt-2 pt-2 border-t border-gray-200">
                                                                <div className="flex flex-wrap gap-1">
                                                                    { node.warnings.map( ( warning, idx ) => (
                                                                        <span
                                                                            key={ idx }
                                                                            className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-700 font-medium"
                                                                        >
                                                                            { warning }
                                                                        </span>
                                                                    ) ) }
                                                                </div>
                                                            </div>
                                                        ) }
                                                    </div>
                                                );
                                            } )
                                        ) }
                                    </div>
                                </div>
                            </div>
                        ) }
                    </div>
                </>
            ) }

            { !comparisonResult && !loading && (
                <div className="text-center text-gray-500 py-12 bg-white rounded-xl shadow-lg">
                    <p className="text-lg">Configure and run analysis to see affected nodes</p>
                </div>
            ) }
        </div>
    );
};

export default Comparison;

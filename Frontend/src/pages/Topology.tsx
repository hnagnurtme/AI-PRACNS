// src/pages/Topology.tsx
import React, { useState, useMemo } from 'react';
import { useNetworkTopology } from '../hooks/useNetworkTopology';
import { useNodeStore } from '../state/nodeStore';
import NetworkCard from '../components/topology/NetworkCard';
import NodeResourceCard from '../components/topology/NodeResourceCard';
import NodeResourceTable from '../components/topology/NodeResourceTable';
import NodeAnalysisModal from '../components/topology/NodeAnalysisModal';
import { getNodeAnalysis } from '../services/nodeAnalysisService';
import type { NodeDTO } from '../types/NodeTypes';
import type { UserTerminal } from '../types/UserTerminalTypes';
import type { NodeAnalysis } from '../types/NodeAnalysisTypes';

const Topology: React.FC = () => {
    const { topology, statistics, loading, error, refetch } = useNetworkTopology();
    const { nodes: realtimeNodes } = useNodeStore(); // Get real-time node updates
    const [ selectedNode, setSelectedNode ] = useState<NodeDTO | null>( null );
    const [ selectedTerminal, setSelectedTerminal ] = useState<UserTerminal | null>( null );
    const [ selectedNodeIds, setSelectedNodeIds ] = useState<Set<string>>( new Set() ); // Filter nodes
    const [ showAllNodes, setShowAllNodes ] = useState<boolean>( true );
    const [ nodeAnalysis, setNodeAnalysis ] = useState<NodeAnalysis | null>( null );
    const [ analysisLoading, setAnalysisLoading ] = useState<boolean>( false );
    const [ analysisError, setAnalysisError ] = useState<Error | null>( null );

    const handleNodeClick = async ( node: NodeDTO ) => {
        setSelectedNode( node );
        setSelectedTerminal( null );

        // Fetch node analysis
        setAnalysisLoading( true );
        setAnalysisError( null );
        try {
            const analysis = await getNodeAnalysis( node.nodeId );
            setNodeAnalysis( analysis );
        } catch ( err ) {
            const error = err instanceof Error ? err : new Error( 'Failed to load node analysis' );
            setAnalysisError( error );
            console.error( 'Failed to load node analysis:', error );
        } finally {
            setAnalysisLoading( false );
        }
    };


    // Merge real-time node updates with topology data
    // Priority: realtimeNodes > topology.nodes (real-time data is more accurate)
    // Deduplicate by nodeId to avoid showing same node twice
    const displayNodes = useMemo( () => {
        const nodeMap = new Map<string, NodeDTO>();

        // First, add topology nodes (base data)
        if ( topology?.nodes ) {
            topology.nodes.forEach( node => {
                nodeMap.set( node.nodeId, node );
            } );
        }

        // Then, override with real-time nodes (more up-to-date)
        if ( realtimeNodes.length > 0 ) {
            realtimeNodes.forEach( node => {
                nodeMap.set( node.nodeId, node );
            } );
        }

        // Return deduplicated array
        return Array.from( nodeMap.values() );
    }, [ topology, realtimeNodes ] );

    // Filter nodes based on selection
    const filteredNodes = showAllNodes
        ? displayNodes
        : displayNodes.filter( node => selectedNodeIds.has( node.nodeId ) );

    if ( loading ) {
        return (
            <div className="flex items-center justify-center h-screen bg-cosmic-black">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-nebula-purple/30 border-t-nebula-purple mx-auto mb-4"></div>
                    <p className="text-star-silver">Loading network topology...</p>
                </div>
            </div>
        );
    }

    if ( error ) {
        return (
            <div className="flex items-center justify-center h-screen bg-cosmic-black">
                <div className="text-center">
                    <div className="text-red-400 text-xl mb-4">⚠️ Error</div>
                    <p className="text-star-silver mb-4">{ error.message }</p>
                    <button
                        onClick={ () => refetch() }
                        className="cosmic-btn"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen w-screen overflow-hidden bg-cosmic-black">
            {/* Sidebar */ }
            <div className="w-96 bg-cosmic-navy/80 backdrop-blur-lg border-r border-white/10 overflow-y-auto p-6 cosmic-scrollbar">
                <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                        <h1 className="text-2xl font-bold bg-gradient-to-r from-nebula-purple via-nebula-pink to-nebula-cyan bg-clip-text text-transparent">Network Topology</h1>
                        <button
                            onClick={ () => refetch() }
                            className="text-xs text-nebula-cyan hover:text-white font-medium px-3 py-1.5 rounded-lg hover:bg-white/10 transition-colors flex items-center gap-1"
                            title="Refresh topology data"
                        >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                            Refresh
                        </button>
                    </div>
                    <p className="text-xs text-star-silver/70 leading-relaxed">
                        Real-time monitoring via WebSocket • Manual refresh available
                    </p>
                    { topology?.updatedAt && (
                        <p className="text-xs text-nebula-cyan mt-1.5 font-medium">
                            ⏱ Updated: { new Date( topology.updatedAt ).toLocaleTimeString() }
                        </p>
                    ) }
                </div>

                { statistics && (
                    <div className="mb-6">
                        <NetworkCard statistics={ statistics } />
                    </div>
                ) }

                {/* Node Filter */ }
                <div className="glass-card p-4 mb-4">
                    <div className="flex justify-between items-center mb-3">
                        <h3 className="font-semibold text-white flex items-center gap-2">
                            <svg className="w-4 h-4 text-nebula-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                            </svg>
                            Node Filter
                        </h3>
                        <button
                            onClick={ () => {
                                setShowAllNodes( !showAllNodes );
                                setSelectedNodeIds( new Set() );
                            } }
                            className="text-xs text-nebula-cyan hover:text-white font-medium px-2 py-1 rounded hover:bg-white/10 transition-colors"
                        >
                            { showAllNodes ? '📋 Select Nodes' : '🌐 Show All' }
                        </button>
                    </div>
                    { !showAllNodes && (
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                            { displayNodes.map( ( node ) => (
                                <label key={ node.nodeId } className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={ selectedNodeIds.has( node.nodeId ) }
                                        onChange={ ( e ) => {
                                            const newSet = new Set( selectedNodeIds );
                                            if ( e.target.checked ) {
                                                newSet.add( node.nodeId );
                                            } else {
                                                newSet.delete( node.nodeId );
                                            }
                                            setSelectedNodeIds( newSet );
                                        } }
                                        className="w-4 h-4 text-nebula-purple border-white/20 rounded focus:ring-nebula-purple bg-white/10"
                                    />
                                    <span className="text-sm text-star-silver">{ node.nodeName }</span>
                                </label>
                            ) ) }
                        </div>
                    ) }
                </div>

                {/* Selected Node Resource Card */ }
                { selectedNode && (
                    <div className="mb-4">
                        <NodeResourceCard node={ selectedNode } />
                        <button
                            onClick={ () => setSelectedNode( null ) }
                            className="mt-2 w-full text-xs text-nebula-cyan hover:text-white text-center"
                        >
                            Clear selection
                        </button>
                    </div>
                ) }

                {/* Selected Terminal Info */ }
                { selectedTerminal && (
                    <div className="glass-card p-4 mb-4">
                        <h3 className="font-semibold text-nebula-pink mb-2">Selected Terminal</h3>
                        <div className="text-sm space-y-1">
                            <div><span className="font-medium">Name:</span> { selectedTerminal.terminalName }</div>
                            <div><span className="font-medium">Type:</span> { selectedTerminal.terminalType }</div>
                            <div><span className="font-medium">Status:</span> { selectedTerminal.status }</div>
                            { selectedTerminal.connectedNodeId && (
                                <div><span className="font-medium">Connected to:</span> { selectedTerminal.connectedNodeId }</div>
                            ) }
                        </div>
                        <button
                            onClick={ () => setSelectedTerminal( null ) }
                            className="mt-2 text-xs text-nebula-cyan hover:text-white"
                        >
                            Clear selection
                        </button>
                    </div>
                ) }

                {/* Legend */ }
                <div className="glass-card p-4">
                    <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                        <svg className="w-4 h-4 text-nebula-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Legend
                    </h3>
                    <div className="space-y-3 text-sm">
                        <div className="font-semibold text-star-silver mb-2 text-xs uppercase tracking-wide">Node Types</div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-cyan-500 rounded-full"></div>
                            <span>LEO Satellite</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                            <span>MEO Satellite</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-purple-500 rounded-full"></div>
                            <span>GEO Satellite</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-orange-500 rounded-full"></div>
                            <span>Ground Station</span>
                        </div>
                        <div className="font-semibold text-star-silver mb-2 mt-4 text-xs uppercase tracking-wide">Health Status</div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-sm"></div>
                            <span className="text-star-silver">Critical (High latency/loss/queue)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-yellow-500 rounded-full border-2 border-white shadow-sm"></div>
                            <span className="text-star-silver">Warning (Elevated metrics)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-lime-500 rounded-full border-2 border-white shadow-sm"></div>
                            <span className="text-star-silver">Normal (Optimal)</span>
                        </div>
                        <div className="font-semibold text-star-silver mb-2 mt-4 text-xs uppercase tracking-wide">Link Status</div>
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-1 bg-lime-500"></div>
                            <span>Active</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-1 bg-yellow-500"></div>
                            <span>Degraded</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-1 bg-red-500"></div>
                            <span>Inactive</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Table and Graph Area */ }
            <div className="flex-1 h-full overflow-auto p-6 cosmic-scrollbar">
                { topology ? (
                    <div className="space-y-6">
                        {/* Node Performance Metrics */ }
                        <div className="glass-card p-6">
                            <div className="flex items-center justify-between mb-6">
                                <div>
                                    <h2 className="text-2xl font-bold bg-gradient-to-r from-nebula-purple via-nebula-pink to-nebula-cyan bg-clip-text text-transparent flex items-center gap-2">
                                        <svg className="w-6 h-6 text-nebula-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={ 2 } d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                        </svg>
                                        Node Performance Metrics
                                    </h2>
                                    <p className="text-sm text-star-silver/70 mt-1">
                                        Real-time monitoring • { filteredNodes.length } nodes displayed
                                    </p>
                                </div>
                            </div>
                            <NodeResourceTable
                                nodes={ filteredNodes }
                                selectedNodeIds={ showAllNodes ? undefined : selectedNodeIds }
                                onNodeClick={ handleNodeClick }
                            />
                        </div>

                    </div>
                ) : (
                    <div className="flex items-center justify-center h-full">
                        <p className="text-star-silver">No topology data available</p>
                    </div>
                ) }
            </div>

            {/* Node Analysis Modal */ }
            { selectedNode && (
                <NodeAnalysisModal
                    node={ selectedNode }
                    analysis={ nodeAnalysis }
                    loading={ analysisLoading }
                    error={ analysisError }
                    onClose={ () => {
                        setSelectedNode( null );
                        setNodeAnalysis( null );
                        setAnalysisError( null );
                    } }
                />
            ) }
        </div>
    );
};

export default Topology;

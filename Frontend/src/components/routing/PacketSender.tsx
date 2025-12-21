import React, { useState, useEffect } from 'react';
import { useTerminalStore } from '../../state/terminalStore';
import { sendPacket } from '../../services/routingService';
import type { RoutingPath, Packet } from '../../types/RoutingTypes';
import type { QoSRequirements } from '../../types/UserTerminalTypes';

interface PacketSenderProps {
    onPathCalculated?: ( path: RoutingPath ) => void;
    onPacketSent?: ( packet: Packet ) => void;
}

// Service Type to QoS mapping
const SERVICE_QOS_MAP: Record<string, QoSRequirements> = {
    VIDEO_STREAM: {
        maxLatencyMs: 150,
        minBandwidthMbps: 20,
        maxLossRate: 0.001,
        priority: 8,
        serviceType: 'VIDEO_STREAM',
    },
    AUDIO_CALL: {
        maxLatencyMs: 100,
        minBandwidthMbps: 0.064,
        maxLossRate: 0.01,
        priority: 9,
        serviceType: 'AUDIO_CALL',
    },
    IMAGE_TRANSFER: {
        maxLatencyMs: 500,
        minBandwidthMbps: 5,
        maxLossRate: 0.0001,
        priority: 6,
        serviceType: 'IMAGE_TRANSFER',
    },
    TEXT_MESSAGE: {
        maxLatencyMs: 1000,
        minBandwidthMbps: 0.1,
        maxLossRate: 0.05,
        priority: 4,
        serviceType: 'TEXT_MESSAGE',
    },
    FILE_TRANSFER: {
        maxLatencyMs: 2000,
        minBandwidthMbps: 10,
        maxLossRate: 0.0001,
        priority: 5,
        serviceType: 'FILE_TRANSFER',
    },
};

const PacketSender: React.FC<PacketSenderProps> = ( { onPathCalculated, onPacketSent } ) => {
    const { terminals } = useTerminalStore();
    const [ sourceTerminalId, setSourceTerminalId ] = useState<string>( '' );
    const [ destinationTerminalId, setDestinationTerminalId ] = useState<string>( '' );
    const [ packetSize, setPacketSize ] = useState<number>( 1024 );
    const [ priority, setPriority ] = useState<number>( 5 );
    const [ routingAlgorithm, setRoutingAlgorithm ] = useState<'dijkstra' | 'rl'>( 'rl' );
    const [ useServiceQos, setUseServiceQos ] = useState<boolean>( false );
    const [ serviceQos, setServiceQos ] = useState<QoSRequirements>( {
        maxLatencyMs: 100,
        minBandwidthMbps: 10,
        maxLossRate: 0.01,
        priority: 5,
        serviceType: 'VIDEO_STREAM',
    } );

    // Track previous service type to detect changes
    const [ previousServiceType, setPreviousServiceType ] = useState<string>( serviceQos.serviceType || 'VIDEO_STREAM' );

    // Auto-update QoS when service type changes (only when user selects a new type)
    useEffect( () => {
        if ( useServiceQos && serviceQos.serviceType && serviceQos.serviceType !== previousServiceType ) {
            const defaultQos = SERVICE_QOS_MAP[ serviceQos.serviceType ];
            if ( defaultQos ) {
                setServiceQos( defaultQos );
                setPreviousServiceType( serviceQos.serviceType );
            }
        }
    }, [ serviceQos.serviceType, useServiceQos, previousServiceType ] );
    const [ loading, setLoading ] = useState<boolean>( false );
    const [ error, setError ] = useState<string | null>( null );
    const [ currentPath, setCurrentPath ] = useState<RoutingPath | null>( null );
    const [ lastPacket, setLastPacket ] = useState<Packet | null>( null );

    // handleCalculatePath removed - only using Send which calculates path automatically

    const handleSendPacket = async () => {
        if ( !sourceTerminalId || !destinationTerminalId ) {
            setError( 'Please select both source and destination terminals' );
            return;
        }

        setLoading( true );
        setError( null );

        try {
            const packet = await sendPacket( {
                sourceTerminalId,
                destinationTerminalId,
                packetSize,
                priority,
                serviceQos: useServiceQos ? serviceQos : undefined,
                algorithm: routingAlgorithm,
            } );
            setLastPacket( packet );
            setCurrentPath( packet.path );
            if ( onPacketSent ) {
                onPacketSent( packet );
            }
            if ( onPathCalculated ) {
                onPathCalculated( packet.path );
            }
        } catch ( err ) {
            setError( err instanceof Error ? err.message : 'Failed to send packet' );
        } finally {
            setLoading( false );
        }
    };


    return (
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg shadow-2xl border border-slate-700 p-4 w-96 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4 pb-3 border-b border-slate-700">
                <h3 className="text-base font-semibold text-white">Send Packet</h3>
            </div>


            <div className="space-y-3">
                <>
                    {/* Source Terminal */ }
                    <div>
                        <label className="block text-xs font-medium text-slate-300 mb-1.5">
                            Source
                        </label>
                        <select
                            value={ sourceTerminalId }
                            onChange={ ( e ) => setSourceTerminalId( e.target.value ) }
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                        >
                            <option value="">Select source...</option>
                            { terminals.map( ( terminal ) => (
                                <option key={ terminal.terminalId } value={ terminal.terminalId }>
                                    { terminal.terminalName }
                                </option>
                            ) ) }
                        </select>
                    </div>

                    {/* Destination Terminal */ }
                    <div>
                        <label className="block text-xs font-medium text-slate-300 mb-1.5">
                            Destination
                        </label>
                        <select
                            value={ destinationTerminalId }
                            onChange={ ( e ) => setDestinationTerminalId( e.target.value ) }
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                        >
                            <option value="">Select destination...</option>
                            { terminals
                                .filter( ( t ) => t.terminalId !== sourceTerminalId )
                                .map( ( terminal ) => (
                                    <option key={ terminal.terminalId } value={ terminal.terminalId }>
                                        { terminal.terminalName }
                                    </option>
                                ) ) }
                        </select>
                    </div>

                    {/* Packet Config - Compact */ }
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">
                                Size (KB)
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="1024"
                                value={ packetSize }
                                onChange={ ( e ) => setPacketSize( parseInt( e.target.value ) || 1024 ) }
                                className="w-full px-2 py-1.5 text-sm bg-slate-700 border border-slate-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">
                                Priority
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="10"
                                value={ priority }
                                onChange={ ( e ) => setPriority( parseInt( e.target.value ) || 5 ) }
                                className="w-full px-2 py-1.5 text-sm bg-slate-700 border border-slate-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>
                    </div>

                    {/* Routing Algorithm */ }
                    <div>
                        <label className="block text-xs font-medium text-slate-300 mb-1.5">
                            Algorithm
                        </label>
                        <select
                            value={ routingAlgorithm }
                            onChange={ ( e ) => setRoutingAlgorithm( e.target.value as 'dijkstra' | 'rl' ) }
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                        >
                            <option value="dijkstra">Dijkstra</option>
                            <option value="rl">RL Agent</option>
                        </select>
                    </div>
                </>


                {/* Use Service QoS */ }
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="useServiceQos"
                        checked={ useServiceQos }
                        onChange={ ( e ) => setUseServiceQos( e.target.checked ) }
                        className="w-4 h-4 text-blue-600 border-slate-500 rounded focus:ring-blue-500 bg-slate-700"
                    />
                    <label htmlFor="useServiceQos" className="text-sm font-medium text-slate-300">
                        Use Service QoS Requirements
                    </label>
                </div>

                {/* Service QoS Settings */ }
                { useServiceQos && (
                    <div className="bg-slate-700/50 border border-slate-600 rounded p-3 space-y-3">
                        <div className="font-semibold text-sm text-slate-200">QoS Requirements</div>

                        {/* Service Type */ }
                        <div>
                            <label className="block text-xs font-medium text-slate-300 mb-1">
                                Service Type (Auto-fills QoS)
                            </label>
                            <select
                                value={ serviceQos.serviceType || 'VIDEO_STREAM' }
                                onChange={ ( e ) => {
                                    const newServiceType = e.target.value;
                                    const defaultQos = SERVICE_QOS_MAP[ newServiceType ];
                                    if ( defaultQos ) {
                                        setServiceQos( defaultQos );
                                        setPreviousServiceType( newServiceType );
                                    } else {
                                        setServiceQos( { ...serviceQos, serviceType: newServiceType } );
                                        setPreviousServiceType( newServiceType );
                                    }
                                } }
                                className="w-full px-2 py-1.5 text-sm border border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
                            >
                                <option value="VIDEO_STREAM">Video Stream (Low Latency, High Bandwidth)</option>
                                <option value="AUDIO_CALL">Audio Call (Very Low Latency)</option>
                                <option value="IMAGE_TRANSFER">Image Transfer (Low Loss Rate)</option>
                                <option value="TEXT_MESSAGE">Text Message (Low Priority)</option>
                                <option value="FILE_TRANSFER">File Transfer (High Bandwidth, Low Loss)</option>
                            </select>
                            <div className="mt-1 text-xs text-slate-400 italic">
                                QoS parameters will be auto-filled based on service type
                            </div>
                        </div>

                        {/* Max Latency */ }
                        <div>
                            <label className="block text-xs font-medium text-slate-300 mb-1">
                                Max Latency (ms)
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="10000"
                                value={ serviceQos.maxLatencyMs }
                                onChange={ ( e ) =>
                                    setServiceQos( {
                                        ...serviceQos,
                                        maxLatencyMs: parseFloat( e.target.value ) || 100,
                                    } )
                                }
                                className="w-full px-2 py-1.5 text-sm border border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
                            />
                            <div className="mt-0.5 text-xs text-slate-400">
                                { serviceQos.serviceType === 'AUDIO_CALL' && 'Real-time: &lt;100ms recommended' }
                                { serviceQos.serviceType === 'VIDEO_STREAM' && 'Streaming: &lt;150ms recommended' }
                                { serviceQos.serviceType === 'TEXT_MESSAGE' && 'Non-critical: &lt;1000ms acceptable' }
                            </div>
                        </div>

                        {/* Min Bandwidth */ }
                        <div>
                            <label className="block text-xs font-medium text-slate-300 mb-1">
                                Min Bandwidth (Mbps)
                            </label>
                            <input
                                type="number"
                                min="0.01"
                                max="1000"
                                step="0.01"
                                value={ serviceQos.minBandwidthMbps }
                                onChange={ ( e ) =>
                                    setServiceQos( {
                                        ...serviceQos,
                                        minBandwidthMbps: parseFloat( e.target.value ) || 10,
                                    } )
                                }
                                className="w-full px-2 py-1.5 text-sm border border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
                            />
                            <div className="mt-0.5 text-xs text-slate-400">
                                { serviceQos.serviceType === 'VIDEO_STREAM' && 'HD Video: ~20Mbps' }
                                { serviceQos.serviceType === 'AUDIO_CALL' && 'Voice: ~0.064Mbps' }
                                { serviceQos.serviceType === 'FILE_TRANSFER' && 'Large files: ~10Mbps' }
                            </div>
                        </div>

                        {/* Max Loss Rate */ }
                        <div>
                            <label className="block text-xs font-medium text-slate-300 mb-1">
                                Max Loss Rate (0-1)
                            </label>
                            <input
                                type="number"
                                min="0"
                                max="1"
                                step="0.0001"
                                value={ serviceQos.maxLossRate }
                                onChange={ ( e ) =>
                                    setServiceQos( {
                                        ...serviceQos,
                                        maxLossRate: parseFloat( e.target.value ) || 0.01,
                                    } )
                                }
                                className="w-full px-2 py-1.5 text-sm border border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
                            />
                            <div className="mt-0.5 text-xs text-slate-400">
                                { serviceQos.serviceType === 'IMAGE_TRANSFER' && 'Critical: &lt;0.01% loss' }
                                { serviceQos.serviceType === 'FILE_TRANSFER' && 'Critical: &lt;0.01% loss' }
                                { serviceQos.serviceType === 'TEXT_MESSAGE' && 'Tolerant: &lt;5% loss acceptable' }
                            </div>
                        </div>

                        {/* QoS Priority */ }
                        <div>
                            <label className="block text-xs font-medium text-slate-300 mb-1">
                                QoS Priority (1-10)
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="10"
                                value={ serviceQos.priority }
                                onChange={ ( e ) =>
                                    setServiceQos( {
                                        ...serviceQos,
                                        priority: parseInt( e.target.value ) || 5,
                                    } )
                                }
                                className="w-full px-2 py-1.5 text-sm border border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
                            />
                            <div className="mt-0.5 text-xs text-slate-400">
                                { serviceQos.priority >= 8 && 'High Priority (Real-time services)' }
                                { serviceQos.priority >= 5 && serviceQos.priority < 8 && 'Medium Priority' }
                                { serviceQos.priority < 5 && 'Low Priority (Best-effort)' }
                            </div>
                        </div>
                    </div>
                ) }

                {/* Error Message */ }
                { error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-3 py-2 rounded text-sm">
                        { error }
                    </div>
                ) }

                {/* Path Info */ }
                { currentPath && (
                    <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm">
                        <div className="font-semibold text-blue-800 mb-2">Path Information</div>
                        <div className="space-y-1 text-blue-700">
                            <div>Hops: { currentPath.hops }</div>
                            <div>Distance: { currentPath.totalDistance.toFixed( 2 ) } km</div>
                            <div>Estimated Latency: { currentPath.estimatedLatency.toFixed( 2 ) } ms</div>
                        </div>
                    </div>
                ) }

                {/* Last Packet Info */ }
                { lastPacket && (
                    <div className="bg-green-50 border border-green-200 rounded p-3 text-sm">
                        <div className="font-semibold text-green-800 mb-2">Packet Sent</div>
                        <div className="space-y-1 text-green-700">
                            <div>Packet ID: { lastPacket.packetId }</div>
                            <div>Status: { lastPacket.status }</div>
                            { lastPacket.serviceQos && (
                                <div className="mt-2 pt-2 border-t border-green-200">
                                    <div className="font-medium">QoS: { lastPacket.serviceQos.serviceType }</div>
                                    <div className="text-xs">
                                        Latency: &lt;{ lastPacket.serviceQos.maxLatencyMs }ms, Bandwidth: &gt;
                                        { lastPacket.serviceQos.minBandwidthMbps }Mbps
                                    </div>
                                </div>
                            ) }
                            { lastPacket.estimatedArrival && (
                                <div>
                                    ETA:{ ' ' }
                                    { new Date( lastPacket.estimatedArrival ).toLocaleTimeString() }
                                </div>
                            ) }
                        </div>
                    </div>
                ) }

                {/* Actions - Only Send Button */ }
                { (
                    <div className="flex gap-2">
                        <button
                            onClick={ handleSendPacket }
                            disabled={ loading || !sourceTerminalId || !destinationTerminalId }
                            className={ `w-full px-4 py-2.5 rounded-md font-medium text-sm transition-colors ${ loading || !sourceTerminalId || !destinationTerminalId
                                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                : 'bg-green-600 text-white hover:bg-green-700'
                                }` }
                        >
                            { loading ? 'Sending...' : '📤 Send Packet' }
                        </button>
                    </div>
                ) }
            </div>
        </div>
    );
};

export default PacketSender;


// 1. Thêm 'useRef' và 'useState', 'useEffect'
import React, { useRef, useState, useEffect } from 'react';
import Draggable from 'react-draggable';
import { useNodeStore } from '../../state/nodeStore';
import type { NodeDTO } from '../../types/NodeTypes';

// --- Component phụ cho Tab Button ---
interface TabButtonProps {
    label: string;
    isActive: boolean;
    onClick: () => void;
}

const TabButton: React.FC<TabButtonProps> = ( { label, isActive, onClick } ) => (
    <button
        onClick={ onClick }
        className={ `px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
            isActive
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
        }` }
    >
        { label }
    </button>
);

// --- Component phụ trợ (không đổi) ---
const DetailItem: React.FC<{ label: string; value: string }> = ( { label, value } ) => (
    <div>
        <dt className="text-gray-500">{ label }</dt>
        <dd className="font-medium text-gray-900">{ value }</dd>
    </div>
);

// --- Component chính ---
interface NodeDetailCardProps {
    node: NodeDTO;
}

type TabName = 'overview' | 'orbital' | 'comms';

const NodeDetailCard: React.FC<NodeDetailCardProps> = ( { node } ) => {
    const { setSelectedNode, cameraFollowMode, setCameraFollowMode } = useNodeStore();
    const nodeRef = useRef( null );

    // 2. Thêm state để quản lý tab
    const [activeTab, setActiveTab] = useState<TabName>( 'overview' );

    // 3. Reset về tab 'overview' khi node được chọn thay đổi
    useEffect( () => {
        setActiveTab( 'overview' );
    }, [node.id] ); // Dùng node.id hoặc node.nodeId

    const handleCameraToggle = () => {
        setCameraFollowMode( !cameraFollowMode );
    };

    // --- Render nội dung của tab ---
    const renderTabContent = () => {
        switch ( activeTab ) {
            case 'overview':
                return (
                    <>
                        {/* --- Core Details --- */ }
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">📊 Trạng thái Node</h4>
                        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                            <DetailItem label="Latitude" value={ node.position.latitude.toFixed( 4 ) } />
                            <DetailItem label="Longitude" value={ node.position.longitude.toFixed( 4 ) } />
                            <DetailItem label="Altitude (Km)" value={ node.position.altitude.toFixed( 2 ) } />

                            { node.batteryChargePercent !== undefined && (
                                <DetailItem label="Battery" value={ `${ node.batteryChargePercent.toFixed( 1 ) }%` } />
                            ) }
                            { node.nodeProcessingDelayMs !== undefined && (
                                <DetailItem label="Processing Delay" value={ `${ node.nodeProcessingDelayMs.toFixed( 2 ) } ms` } />
                            ) }
                            { node.packetLossRate !== undefined && (
                                <DetailItem label="Loss Rate" value={ ( node.packetLossRate * 100 ).toFixed( 3 ) + '%' } />
                            ) }
                            { node.resourceUtilization !== undefined && (
                                <DetailItem label="Resource Usage" value={ `${ node.resourceUtilization.toFixed( 1 ) }%` } />
                            ) }
                            { node.packetBufferCapacity !== undefined && node.currentPacketCount !== undefined && (
                                <DetailItem label="Buffer" value={ `${ node.currentPacketCount }/${ node.packetBufferCapacity }` } />
                            ) }
                            { node.weather && (
                                <DetailItem label="Weather" value={ node.weather.replace( /_/g, ' ' ) } />
                            ) }
                            { node.host && node.port && (
                                <DetailItem label="Host:Port" value={ `${ node.host }:${ node.port }` } />
                            ) }
                        </div>

                        {/* --- Last Updated --- */ }
                        { node.lastUpdated && (
                            <div className="mt-4 pt-3 border-t border-gray-100">
                                <h4 className="text-sm font-semibold text-gray-700 mb-1">Cập nhật lần cuối</h4>
                                <div className="text-xs text-gray-500">
                                    { new Date( node.lastUpdated ).toLocaleString() }
                                </div>
                            </div>
                        ) }
                    </>
                );
            case 'orbital':
                return (
                    <>
                        {/* --- Orbital Information --- */ }
                        { ( node.orbit || node.velocity ) && (
                            <>
                                <h4 className="text-sm font-semibold text-gray-700 mb-2">🛰️ Dữ liệu Quỹ đạo</h4>
                                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                                    { node.orbit && (
                                        <>
                                            <DetailItem label="Semi-major Axis" value={ `${ node.orbit.semiMajorAxisKm.toFixed( 0 ) } km` } />
                                            <DetailItem label="Eccentricity" value={ node.orbit.eccentricity.toFixed( 4 ) } />
                                            <DetailItem label="Inclination" value={ `${ node.orbit.inclinationDeg.toFixed( 2 ) }°` } />
                                            <DetailItem label="RAAN" value={ `${ node.orbit.raanDeg.toFixed( 2 ) }°` } />
                                            <DetailItem label="Arg. of Perigee" value={ `${ node.orbit.argumentOfPerigeeDeg.toFixed( 2 ) }°` } />
                                            <DetailItem label="True Anomaly" value={ `${ node.orbit.trueAnomalyDeg.toFixed( 2 ) }°` } />
                                        </>
                                    ) }
                                    { node.velocity && (
                                        <>
                                            <DetailItem label="Velocity X" value={ `${ node.velocity.velocityX.toFixed( 2 ) } km/s` } />
                                            <DetailItem label="Velocity Y" value={ `${ node.velocity.velocityY.toFixed( 2 ) } km/s` } />
                                            <DetailItem label="Velocity Z" value={ `${ node.velocity.velocityZ.toFixed( 2 ) } km/s` } />
                                        </>
                                    ) }
                                </div>
                            </>
                        ) }
                    </>
                );
            case 'comms':
                return (
                    <>
                        {/* --- Communication Details --- */ }
                        { node.communication && (
                            <>
                                <h4 className="text-sm font-semibold text-gray-700 mb-2">📡 Dữ liệu Liên lạc</h4>
                                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                                    <DetailItem label="Protocol" value={ node.communication.protocol } />
                                    <DetailItem label="Frequency" value={ `${ node.communication.frequencyGHz } GHz` } />
                                    <DetailItem label="Bandwidth" value={ `${ node.communication.bandwidthMHz } MHz` } />
                                    <DetailItem label="Transmit Power" value={ `${ node.communication.transmitPowerDbW } dBW` } />
                                    <DetailItem label="Antenna Gain" value={ `${ node.communication.antennaGainDb } dB` } />
                                    <DetailItem label="Beam Width" value={ `${ node.communication.beamWidthDeg }°` } />
                                    <DetailItem label="Max Range" value={ `${ node.communication.maxRangeKm } km` } />
                                    <DetailItem label="Min Elevation" value={ `${ node.communication.minElevationDeg }°` } />
                                    <DetailItem label="IP Address" value={ node.communication.ipAddress } />
                                    <DetailItem label="Comm Port" value={ `${ node.communication.port }` } />
                                </div>
                            </>
                        ) }
                    </>
                );
            default:
                return null;
        }
    };

    return (
        <Draggable
            handle=".drag-handle"
            bounds="parent"
            nodeRef={ nodeRef }
        >
            {/* 4. Gán ref và thay đổi kích thước card */ }
            <div
                ref={ nodeRef }
                className="absolute top-20 right-4 z-50 w-[450px] 
                           bg-white rounded-xl shadow-xl border border-gray-200
                           flex flex-col" // Thêm flex-col
            >
                {/* --- Tay nắm để kéo thả (Header) --- */ }
                <div className="drag-handle flex justify-between items-start p-4 pb-2 cursor-move">
                    <div className="flex-1 truncate">
                        <h3 className="text-xl font-extrabold text-gray-800 truncate select-none">
                            { node.nodeName }
                        </h3>
                        <p className="text-xs text-gray-500 select-none">
                            ID: { node.nodeId.substring( 0, 8 ) }...
                        </p>
                    </div>
                    <button
                        onClick={ () => setSelectedNode( null ) }
                        className="text-gray-500 hover:text-red-600 text-lg font-bold ml-2"
                    >
                        &times;
                    </button>
                </div>

                {/* --- Status Badges --- */ }
                <div className="px-4 pb-3 border-b border-gray-200">
                    <p className="text-sm flex flex-wrap gap-2 items-center">
                        <span className={ `px-2 py-0.5 rounded-full text-xs font-semibold ${ node.isOperational ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600' }` }>
                            { node.isOperational ? 'OPERATIONAL' : 'OFFLINE' }
                        </span>
                        <span className={ `px-2 py-0.5 rounded-full text-xs font-semibold ${ node.healthy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700' }` }>
                            { node.healthy ? 'HEALTHY' : 'UNHEALTHY' }
                        </span>
                        <span className="text-gray-500">({ node.nodeType })</span>
                        { cameraFollowMode && (
                            <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-700 animate-pulse">
                                📹 ĐANG THEO DÕI
                            </span>
                        ) }
                    </p>
                </div>

                {/* --- 5. Thanh điều hướng Tab --- */ }
                <div className="flex border-b border-gray-200">
                    <TabButton
                        label="Tổng quan"
                        isActive={ activeTab === 'overview' }
                        onClick={ () => setActiveTab( 'overview' ) }
                    />
                    
                    {/* Chỉ hiển thị tab nếu có dữ liệu */ }
                    { ( node.orbit || node.velocity ) && (
                        <TabButton
                            label="Quỹ đạo"
                            isActive={ activeTab === 'orbital' }
                            onClick={ () => setActiveTab( 'orbital' ) }
                        />
                    ) }
                    { node.communication && (
                        <TabButton
                            label="Liên lạc"
                            isActive={ activeTab === 'comms' }
                            onClick={ () => setActiveTab( 'comms' ) }
                        />
                    ) }
                </div>

                {/* --- 6. Nội dung Tab (với scroll) --- */ }
                <div className="p-4 bg-gray-50 max-h-72 overflow-y-auto">
                    { renderTabContent() }
                </div>

                {/* --- Action Buttons (Footer) --- */ }
                <div className="p-4 flex flex-wrap gap-2 border-t border-gray-200 bg-white rounded-b-xl">
                    <button
                        onClick={ handleCameraToggle }
                        className={ `text-sm font-medium px-3 py-1 rounded border transition-colors ${ cameraFollowMode
                            ? 'bg-green-600 text-white border-green-600 hover:bg-green-700'
                            : 'text-green-600 border-green-200 hover:bg-green-50'
                            }` }
                    >
                        📹 { cameraFollowMode ? 'Đang theo dõi' : 'Theo dõi Camera' }
                    </button>
                </div>
            </div>
        </Draggable>
    );
};

export default NodeDetailCard;
import React, { useRef, useState, useEffect } from 'react';
import Draggable from 'react-draggable';
import { useNodeStore } from '../../state/nodeStore';
import { updateNodeStatus } from '../../services/nodeService';
import type { NodeDTO, UpdateStatusRequest, WeatherType } from '../../types/NodeTypes';

interface TabButtonProps {
    label: string;
    isActive: boolean;
    onClick: () => void;
}

const TabButton: React.FC<TabButtonProps> = ({ label, isActive, onClick }) => (
    <button
        onClick={onClick}
        className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
            isActive
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
        }`}
    >
        {label}
    </button>
);

const DetailItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div>
        <dt className="text-gray-500 text-xs">{label}</dt>
        <dd className="font-medium text-gray-900 text-sm">{value}</dd>
    </div>
);

interface EditableSliderProps {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    unit?: string;
    onChange: (value: number) => void;
}

const EditableSlider: React.FC<EditableSliderProps> = ({ label, value, min, max, step, unit = '', onChange }) => (
    <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
            <label className="text-xs font-medium text-gray-700">{label}</label>
            <span className="text-xs font-semibold text-blue-600">{value.toFixed(1)}{unit}</span>
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
    </div>
);

interface NodeDetailCardProps {
    node: NodeDTO;
}

type TabName = 'overview' | 'edit' | 'orbital' | 'comms';

const NodeDetailCard: React.FC<NodeDetailCardProps> = ({ node }) => {
    const { setSelectedNode, cameraFollowMode, setCameraFollowMode, updateNodeInStore } = useNodeStore();
    const nodeRef = useRef(null);

    const [activeTab, setActiveTab] = useState<TabName>('overview');
    const [isUpdating, setIsUpdating] = useState(false);
    
    // Local state for editable fields
    const [batteryLevel, setBatteryLevel] = useState(node.batteryChargePercent || 100);
    const [processingDelay, setProcessingDelay] = useState(node.nodeProcessingDelayMs || 10);
    const [packetLossRate, setPacketLossRate] = useState((node.packetLossRate || 0) * 100);
    const [resourceUtil, setResourceUtil] = useState(node.resourceUtilization || 50);
    const [currentPackets, setCurrentPackets] = useState(node.currentPacketCount || 0);
    const [weather, setWeather] = useState<WeatherType>(node.weather as WeatherType || 'CLEAR');
    const [isOperational, setIsOperational] = useState(node.isOperational);

    // Reset local state when node changes
    useEffect(() => {
        setActiveTab('overview');
        setBatteryLevel(node.batteryChargePercent || 100);
        setProcessingDelay(node.nodeProcessingDelayMs || 10);
        setPacketLossRate((node.packetLossRate || 0) * 100);
        setResourceUtil(node.resourceUtilization || 50);
        setCurrentPackets(node.currentPacketCount || 0);
        setWeather(node.weather as WeatherType || 'CLEAR');
        setIsOperational(node.isOperational);
    }, [node.id]);

    const handleCameraToggle = () => {
        setCameraFollowMode(!cameraFollowMode);
    };

    const handleApplyChanges = async () => {
        setIsUpdating(true);
        try {
            const updateData: UpdateStatusRequest = {
                batteryChargePercent: batteryLevel,
                nodeProcessingDelayMs: processingDelay,
                packetLossRate: packetLossRate / 100,
                resourceUtilization: resourceUtil,
                currentPacketCount: currentPackets,
                weather: weather,
                isOperational: isOperational
            };

            const updatedNode = await updateNodeStatus(node.id, updateData);
            updateNodeInStore(updatedNode);
            console.log('✅ Node updated successfully:', updatedNode);
        } catch (error) {
            console.error('❌ Failed to update node:', error);
            alert('Failed to update node. Please try again.');
        } finally {
            setIsUpdating(false);
        }
    };

    const renderTabContent = () => {
        switch (activeTab) {
            case 'overview':
                return (
                    <>
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">📊 Trạng thái Node</h4>
                        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                            <DetailItem label="Latitude" value={node.position.latitude.toFixed(4)} />
                            <DetailItem label="Longitude" value={node.position.longitude.toFixed(4)} />
                            <DetailItem label="Altitude (Km)" value={node.position.altitude.toFixed(2)} />
                            {node.batteryChargePercent !== undefined && (
                                <DetailItem label="Battery" value={`${node.batteryChargePercent.toFixed(1)}%`} />
                            )}
                            {node.nodeProcessingDelayMs !== undefined && (
                                <DetailItem label="Processing Delay" value={`${node.nodeProcessingDelayMs.toFixed(2)} ms`} />
                            )}
                            {node.packetLossRate !== undefined && (
                                <DetailItem label="Loss Rate" value={(node.packetLossRate * 100).toFixed(3) + '%'} />
                            )}
                            {node.resourceUtilization !== undefined && (
                                <DetailItem label="Resource Usage" value={`${node.resourceUtilization.toFixed(1)}%`} />
                            )}
                            {node.packetBufferCapacity !== undefined && node.currentPacketCount !== undefined && (
                                <DetailItem label="Buffer" value={`${node.currentPacketCount}/${node.packetBufferCapacity}`} />
                            )}
                            {node.weather && (
                                <DetailItem label="Weather" value={node.weather.replace(/_/g, ' ')} />
                            )}
                            {node.host && node.port && (
                                <DetailItem label="Host:Port" value={`${node.host}:${node.port}`} />
                            )}
                        </div>
                        {node.lastUpdated && (
                            <div className="mt-4 pt-3 border-t border-gray-100">
                                <h4 className="text-sm font-semibold text-gray-700 mb-1">Cập nhật lần cuối</h4>
                                <div className="text-xs text-gray-500">
                                    {new Date(node.lastUpdated).toLocaleString()}
                                </div>
                            </div>
                        )}
                    </>
                );
            
            case 'edit':
                return (
                    <>
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">✏️ Chỉnh sửa thủ công</h4>
                        
                        {/* Operational Status Toggle */}
                        <div className="mb-3">
                            <label className="flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={isOperational}
                                    onChange={(e) => setIsOperational(e.target.checked)}
                                    className="mr-2 w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                                />
                                <span className="text-sm font-medium text-gray-700">Node Operational</span>
                            </label>
                        </div>

                        {/* Battery Level */}
                        <EditableSlider
                            label="Battery Level"
                            value={batteryLevel}
                            min={0}
                            max={100}
                            step={1}
                            unit="%"
                            onChange={setBatteryLevel}
                        />

                        {/* Processing Delay */}
                        <EditableSlider
                            label="Processing Delay"
                            value={processingDelay}
                            min={0}
                            max={500}
                            step={5}
                            unit=" ms"
                            onChange={setProcessingDelay}
                        />

                        {/* Packet Loss Rate */}
                        <EditableSlider
                            label="Packet Loss Rate"
                            value={packetLossRate}
                            min={0}
                            max={100}
                            step={0.1}
                            unit="%"
                            onChange={setPacketLossRate}
                        />

                        {/* Resource Utilization */}
                        <EditableSlider
                            label="Resource Utilization"
                            value={resourceUtil}
                            min={0}
                            max={100}
                            step={1}
                            unit="%"
                            onChange={setResourceUtil}
                        />

                        {/* Current Packets in Buffer */}
                        <EditableSlider
                            label="Current Packets in Buffer"
                            value={currentPackets}
                            min={0}
                            max={node.packetBufferCapacity || 1000}
                            step={1}
                            unit=""
                            onChange={setCurrentPackets}
                        />

                        {/* Weather Condition */}
                        <div className="mb-3">
                            <label className="text-xs font-medium text-gray-700 block mb-1">Weather Condition</label>
                            <select
                                value={weather}
                                onChange={(e) => setWeather(e.target.value as WeatherType)}
                                className="w-full px-2 py-1 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="CLEAR">Clear</option>
                                <option value="LIGHT_RAIN">Light Rain</option>
                                <option value="RAIN">Rain</option>
                                <option value="SNOW">Snow</option>
                                <option value="STORM">Storm</option>
                                <option value="SEVERE_STORM">Severe Storm</option>
                            </select>
                        </div>

                        {/* Apply Button */}
                        <button
                            onClick={handleApplyChanges}
                            disabled={isUpdating}
                            className={`w-full mt-4 px-4 py-2 text-sm font-medium text-white rounded-md transition-colors ${
                                isUpdating
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
                            }`}
                        >
                            {isUpdating ? 'Đang cập nhật...' : 'Áp dụng thay đổi'}
                        </button>
                    </>
                );

            case 'orbital':
                return (
                    <>
                        {(node.orbit || node.velocity) && (
                            <>
                                <h4 className="text-sm font-semibold text-gray-700 mb-2">🛰️ Dữ liệu Quỹ đạo</h4>
                                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                                    {node.orbit && (
                                        <>
                                            <DetailItem label="Semi-major Axis" value={`${node.orbit.semiMajorAxisKm.toFixed(0)} km`} />
                                            <DetailItem label="Eccentricity" value={node.orbit.eccentricity.toFixed(4)} />
                                            <DetailItem label="Inclination" value={`${node.orbit.inclinationDeg.toFixed(2)}°`} />
                                            <DetailItem label="RAAN" value={`${node.orbit.raanDeg.toFixed(2)}°`} />
                                            <DetailItem label="Arg. of Perigee" value={`${node.orbit.argumentOfPerigeeDeg.toFixed(2)}°`} />
                                            <DetailItem label="True Anomaly" value={`${node.orbit.trueAnomalyDeg.toFixed(2)}°`} />
                                        </>
                                    )}
                                    {node.velocity && (
                                        <>
                                            <DetailItem label="Velocity X" value={`${node.velocity.velocityX.toFixed(2)} km/s`} />
                                            <DetailItem label="Velocity Y" value={`${node.velocity.velocityY.toFixed(2)} km/s`} />
                                            <DetailItem label="Velocity Z" value={`${node.velocity.velocityZ.toFixed(2)} km/s`} />
                                        </>
                                    )}
                                </div>
                            </>
                        )}
                    </>
                );

            case 'comms':
                return (
                    <>
                        {node.communication && (
                            <>
                                <h4 className="text-sm font-semibold text-gray-700 mb-2">📡 Dữ liệu Liên lạc</h4>
                                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                                    <DetailItem label="Protocol" value={node.communication.protocol} />
                                    <DetailItem label="Frequency" value={`${node.communication.frequencyGHz} GHz`} />
                                    <DetailItem label="Bandwidth" value={`${node.communication.bandwidthMHz} MHz`} />
                                    <DetailItem label="Transmit Power" value={`${node.communication.transmitPowerDbW} dBW`} />
                                    <DetailItem label="Antenna Gain" value={`${node.communication.antennaGainDb} dB`} />
                                    <DetailItem label="Beam Width" value={`${node.communication.beamWidthDeg}°`} />
                                    <DetailItem label="Max Range" value={`${node.communication.maxRangeKm} km`} />
                                    <DetailItem label="Min Elevation" value={`${node.communication.minElevationDeg}°`} />
                                    <DetailItem label="IP Address" value={node.communication.ipAddress} />
                                    <DetailItem label="Comm Port" value={`${node.communication.port}`} />
                                </div>
                            </>
                        )}
                    </>
                );

            default:
                return null;
        }
    };

    return (
        <Draggable handle=".drag-handle" bounds="parent" nodeRef={nodeRef}>
            <div
                ref={nodeRef}
                className="absolute top-20 right-4 z-50 w-[450px] 
                           bg-white rounded-xl shadow-xl border border-gray-200
                           flex flex-col"
            >
                {/* Header */}
                <div className="drag-handle flex justify-between items-start p-4 pb-2 cursor-move">
                    <div className="flex-1 truncate">
                        <h3 className="text-xl font-extrabold text-gray-800 truncate select-none">
                            {node.nodeName}
                        </h3>
                        <p className="text-xs text-gray-500 select-none">
                            ID: {node.nodeId.substring(0, 8)}...
                        </p>
                    </div>
                    <button
                        onClick={() => setSelectedNode(null)}
                        className="text-gray-500 hover:text-red-600 text-lg font-bold ml-2"
                    >
                        &times;
                    </button>
                </div>

                {/* Status Badges */}
                <div className="px-4 pb-3 border-b border-gray-200">
                    <p className="text-sm flex flex-wrap gap-2 items-center">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${node.isOperational ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}>
                            {node.isOperational ? 'OPERATIONAL' : 'OFFLINE'}
                        </span>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${node.healthy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                            {node.healthy ? 'HEALTHY' : 'UNHEALTHY'}
                        </span>
                        <span className="text-gray-500">({node.nodeType})</span>
                        {cameraFollowMode && (
                            <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-700 animate-pulse">
                                📹 ĐANG THEO DÕI
                            </span>
                        )}
                    </p>
                </div>

                {/* Tab Navigation */}
                <div className="flex border-b border-gray-200">
                    <TabButton
                        label="Tổng quan"
                        isActive={activeTab === 'overview'}
                        onClick={() => setActiveTab('overview')}
                    />
                    <TabButton
                        label="Chỉnh sửa"
                        isActive={activeTab === 'edit'}
                        onClick={() => setActiveTab('edit')}
                    />
                    {(node.orbit || node.velocity) && (
                        <TabButton
                            label="Quỹ đạo"
                            isActive={activeTab === 'orbital'}
                            onClick={() => setActiveTab('orbital')}
                        />
                    )}
                    {node.communication && (
                        <TabButton
                            label="Liên lạc"
                            isActive={activeTab === 'comms'}
                            onClick={() => setActiveTab('comms')}
                        />
                    )}
                </div>

                {/* Tab Content */}
                <div className="p-4 bg-gray-50 max-h-72 overflow-y-auto">
                    {renderTabContent()}
                </div>

                {/* Footer Actions */}
                <div className="p-4 flex flex-wrap gap-2 border-t border-gray-200 bg-white rounded-b-xl">
                    <button
                        onClick={handleCameraToggle}
                        className={`text-sm font-medium px-3 py-1 rounded border transition-colors ${cameraFollowMode
                            ? 'bg-green-600 text-white border-green-600 hover:bg-green-700'
                            : 'text-green-600 border-green-200 hover:bg-green-50'
                        }`}
                    >
                        📹 {cameraFollowMode ? 'Đang theo dõi' : 'Theo dõi Camera'}
                    </button>
                </div>
            </div>
        </Draggable>
    );
};

export default NodeDetailCard;

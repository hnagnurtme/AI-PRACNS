import React, { useState } from 'react';
import { useTerminalStore } from '../../state/terminalStore';
import { sendPacket } from '../../services/routingService';
import type { Packet, RoutingPath } from '../../types/RoutingTypes';

interface SendPacketCardProps {
    onPacketSent?: (packet: Packet) => void;
    onPathCalculated?: (path: RoutingPath) => void;
    onClose?: () => void;
}

const SendPacketCard: React.FC<SendPacketCardProps> = ({ 
    onPacketSent, 
    onPathCalculated,
    onClose 
}) => {
    const { sourceTerminal, destinationTerminal, clearPacketSelection } = useTerminalStore();
    const [packetSize, setPacketSize] = useState<number>(1024);
    const [priority, setPriority] = useState<number>(5);
    const [routingAlgorithm, setRoutingAlgorithm] = useState<'simple' | 'dijkstra' | 'rl'>('rl');
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // Nếu chưa có đủ 2 terminals, không hiển thị
    if (!sourceTerminal || !destinationTerminal) {
        return null;
    }

    const handleSendPacket = async () => {
        if (!sourceTerminal || !destinationTerminal) {
            setError('Please select both source and destination terminals');
            return;
        }

        if (sourceTerminal.terminalId === destinationTerminal.terminalId) {
            setError('Source and destination cannot be the same');
            return;
        }

        setLoading(true);
        setError(null);
        setSuccess(null);

        try {
            const packet = await sendPacket({
                sourceTerminalId: sourceTerminal.terminalId,
                destinationTerminalId: destinationTerminal.terminalId,
                packetSize,
                priority,
                algorithm: routingAlgorithm,
            });
            
            setSuccess(`Packet sent successfully! ID: ${packet.packetId}`);
            
            if (onPacketSent) {
                onPacketSent(packet);
            }
            
            if (onPathCalculated && packet.path) {
                onPathCalculated(packet.path);
            }

            // Clear selection sau 2 giây
            setTimeout(() => {
                clearPacketSelection();
                if (onClose) {
                    onClose();
                }
            }, 2000);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to send packet');
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        clearPacketSelection();
        if (onClose) {
            onClose();
        }
    };

    return (
        <div className="absolute top-20 right-4 z-20 w-96 bg-white rounded-lg shadow-2xl border border-gray-200 p-6 animate-slide-in">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-gray-800">📦 Send Packet</h3>
                <button
                    onClick={handleClear}
                    className="text-gray-500 hover:text-red-600 text-2xl font-bold transition-colors"
                    title="Close"
                >
                    &times;
                </button>
            </div>

            {/* Source & Destination Info */}
            <div className="mb-4 space-y-2">
                <div className="flex items-center gap-2 p-2 bg-green-50 rounded-lg border border-green-200">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <div className="flex-1">
                        <div className="text-xs text-gray-500">Source</div>
                        <div className="font-semibold text-gray-800">{sourceTerminal.terminalName}</div>
                    </div>
                </div>
                <div className="flex items-center gap-2 p-2 bg-red-50 rounded-lg border border-red-200">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <div className="flex-1">
                        <div className="text-xs text-gray-500">Destination</div>
                        <div className="font-semibold text-gray-800">{destinationTerminal.terminalName}</div>
                    </div>
                </div>
            </div>

            {/* Packet Size */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                    Packet Size (bytes)
                </label>
                <input
                    type="number"
                    min="1"
                    max="10485760" // 10MB
                    value={packetSize}
                    onChange={(e) => setPacketSize(parseInt(e.target.value) || 1024)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <div className="text-xs text-gray-500 mt-1">
                    {(packetSize / 1024).toFixed(2)} KB
                </div>
            </div>

            {/* Priority */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                    Priority: {priority}
                </label>
                <input
                    type="range"
                    min="1"
                    max="10"
                    value={priority}
                    onChange={(e) => setPriority(parseInt(e.target.value))}
                    className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Low (1)</span>
                    <span>High (10)</span>
                </div>
            </div>

            {/* Routing Algorithm */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Routing Algorithm
                </label>
                <div className="flex gap-2">
                    {(['simple', 'dijkstra', 'rl'] as const).map((algo) => (
                        <button
                            key={algo}
                            onClick={() => setRoutingAlgorithm(algo)}
                            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                                routingAlgorithm === algo
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                        >
                            {algo.toUpperCase()}
                        </button>
                    ))}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                    ⚠️ {error}
                </div>
            )}

            {/* Success Message */}
            {success && (
                <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700">
                    ✅ {success}
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-2">
                <button
                    onClick={handleSendPacket}
                    disabled={loading}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-4 py-2 rounded-lg font-medium hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? 'Sending...' : '📤 Send Packet'}
                </button>
                <button
                    onClick={handleClear}
                    className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                >
                    Clear
                </button>
            </div>

            {/* Hint */}
            <div className="mt-4 text-xs text-gray-500 text-center">
                💡 Click on terminals on the map to select source and destination
            </div>
        </div>
    );
};

export default SendPacketCard;


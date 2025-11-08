import React from 'react';
import type { HopRecord } from '../../types/ComparisonTypes';

interface HopTooltipProps {
    hop: HopRecord;
    index: number;
}

export const HopTooltip: React.FC<HopTooltipProps> = ({ hop, index }) => {
    return (
        <div className="bg-white rounded-lg shadow-lg p-4 max-w-md border border-gray-200">
            <div className="font-semibold text-lg mb-3 text-gray-800 border-b pb-2">
                Hop #{index + 1}
            </div>
            
            <div className="space-y-2 text-sm">
                <div className="grid grid-cols-2 gap-2">
                    <div className="text-gray-600">From:</div>
                    <div className="font-medium text-gray-900">{hop.fromNodeId}</div>
                    
                    <div className="text-gray-600">To:</div>
                    <div className="font-medium text-gray-900">{hop.toNodeId}</div>
                </div>
                
                <div className="border-t pt-2 mt-2">
                    <div className="grid grid-cols-2 gap-2">
                        <div className="text-gray-600">Latency:</div>
                        <div className="font-medium text-blue-600">{hop.latencyMs.toFixed(2)} ms</div>
                        
                        <div className="text-gray-600">Distance:</div>
                        <div className="font-medium text-blue-600">{hop.distanceKm.toFixed(2)} km</div>
                        
                        <div className="text-gray-600">Timestamp:</div>
                        <div className="font-medium text-gray-700">
                            {new Date(hop.timestampMs).toLocaleTimeString()}
                        </div>
                    </div>
                </div>

                {hop.routingDecisionInfo && (
                    <div className="border-t pt-2 mt-2">
                        <div className="text-gray-600 mb-1">Routing Algorithm:</div>
                        <div className="font-medium text-purple-600">
                            {hop.routingDecisionInfo.algorithm}
                        </div>
                        {hop.routingDecisionInfo.reward !== undefined && (
                            <div className="text-xs text-gray-500 mt-1">
                                Reward: {hop.routingDecisionInfo.reward.toFixed(4)}
                            </div>
                        )}
                    </div>
                )}

                {hop.fromNodeBufferState && (
                    <div className="border-t pt-2 mt-2">
                        <div className="text-gray-600 mb-1">Buffer State:</div>
                        <div className="grid grid-cols-2 gap-2">
                            <div className="text-xs text-gray-600">Queue Size:</div>
                            <div className="text-xs font-medium text-gray-900">
                                {hop.fromNodeBufferState.queueSize}
                            </div>
                            
                            <div className="text-xs text-gray-600">Bandwidth Util:</div>
                            <div className="text-xs font-medium text-gray-900">
                                {(hop.fromNodeBufferState.bandwidthUtilization * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                )}

                {hop.scenarioType && hop.scenarioType !== 'NORMAL' && (
                    <div className="border-t pt-2 mt-2 bg-yellow-50 -m-4 p-4 rounded-b-lg">
                        <div className="flex items-center">
                            <svg className="h-5 w-5 text-yellow-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                            <div>
                                <div className="text-xs font-semibold text-yellow-800">
                                    Scenario: {hop.scenarioType.replace(/_/g, ' ')}
                                </div>
                                {hop.nodeLoadPercent !== undefined && (
                                    <div className="text-xs text-yellow-700 mt-1">
                                        Node Load: {hop.nodeLoadPercent.toFixed(1)}%
                                    </div>
                                )}
                                {hop.dropReasonDetails && (
                                    <div className="text-xs text-red-600 mt-1">
                                        Drop Reason: {hop.dropReasonDetails}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// src/hooks/useBatchPolling.ts

import { useEffect, useState, useRef } from "react";
import type { NetworkBatch } from "../types/ComparisonTypes";
import axiosClient from "../api/axiosClient";

type ConnectionStatus = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED';

interface UseBatchPollingResult {
    receivedBatches: NetworkBatch[];
    connectionStatus: ConnectionStatus;
}

const POLL_INTERVAL = 1500; // Poll every 1.5 seconds
const BATCH_ENDPOINT = '/api/v1/batch/poll';

/**
 * Custom Hook để poll batches từ backend endpoint
 */
export const useBatchPolling = (enabled: boolean = true): UseBatchPollingResult => {
    const [receivedBatches, setReceivedBatches] = useState<NetworkBatch[]>([]);
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('DISCONNECTED');
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const isPollingRef = useRef(false);
    const statusRef = useRef<ConnectionStatus>('DISCONNECTED');

    useEffect(() => {
        if (!enabled) {
            setConnectionStatus('DISCONNECTED');
            statusRef.current = 'DISCONNECTED';
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            return;
        }

        setConnectionStatus('CONNECTING');
        statusRef.current = 'CONNECTING';
        isPollingRef.current = true;

        const pollBatch = async () => {
            if (!isPollingRef.current) return;

            try {
                const response = await axiosClient.get<NetworkBatch | { status: string; message: string }>(
                    BATCH_ENDPOINT
                );

                // Check if we got a batch (status 200) or no data (status 204)
                if (response.status === 200 && 'batchId' in response.data) {
                    const batch = response.data as NetworkBatch;
                    
                    // Check if this batch is new (not already in receivedBatches)
                    setReceivedBatches((prev) => {
                        const exists = prev.some(b => b.batchId === batch.batchId);
                        if (!exists) {
                            console.log("📩 New batch received:", batch.batchId);
                            return [batch, ...prev];
                        }
                        return prev;
                    });

                    // Update status only if it changed
                    if (statusRef.current !== 'CONNECTED') {
                        statusRef.current = 'CONNECTED';
                        setConnectionStatus('CONNECTED');
                    }
                } else if (response.status === 204) {
                    // No data available, but connection is working
                    if (statusRef.current !== 'CONNECTED') {
                        statusRef.current = 'CONNECTED';
                        setConnectionStatus('CONNECTED');
                    }
                }
            } catch (error: any) {
                console.error("❌ Error polling batch:", error);
                
                // If it's a network error, set to disconnected
                if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
                    statusRef.current = 'DISCONNECTED';
                    setConnectionStatus('DISCONNECTED');
                } else if (error.response?.status === 404) {
                    // Endpoint not found
                    statusRef.current = 'DISCONNECTED';
                    setConnectionStatus('DISCONNECTED');
                } else {
                    // Other errors, but connection might still be working
                    if (statusRef.current === 'CONNECTING') {
                        statusRef.current = 'CONNECTED';
                        setConnectionStatus('CONNECTED');
                    }
                }
            }
        };

        // Start polling immediately
        pollBatch();

        // Set up interval for polling
        intervalRef.current = setInterval(pollBatch, POLL_INTERVAL);

        return () => {
            isPollingRef.current = false;
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            statusRef.current = 'DISCONNECTED';
            setConnectionStatus('DISCONNECTED');
        };
    }, [enabled]); // Removed connectionStatus from dependencies

    return { receivedBatches, connectionStatus };
};


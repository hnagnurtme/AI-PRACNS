// src/components/LatencyDistChart.tsx
import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import type { LatencyDistData } from '../../types/type';

// Định nghĩa Props cho component này
interface LatencyDistChartProps {
    data: LatencyDistData[];
}

export const LatencyDistChart: React.FC<LatencyDistChartProps> = ( { data } ) => {
    return (
        <div className="bg-white rounded-xl shadow-lg p-4 md:p-6 transition-all duration-300 hover:shadow-2xl flex flex-col">
            <h3 className="text-center text-xl font-semibold text-slate-700 mb-4">
                Packet Latency Distribution (Jitter)
            </h3>
            <div className="flex-1 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={ data }>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={ { fontSize: 12 } } />
                        <YAxis tick={ { fontSize: 12 } } />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="RL" fill="#8884d8" name="RL (Packet Count)" />
                        <Bar dataKey="NonRL" fill="#82ca9d" name="Non-RL (Packet Count)" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            <p className="text-center text-sm text-slate-500 mt-3">
                RL is stable (clustered). Non-RL is spread towards higher latency.
            </p>
        </div>
    );
};
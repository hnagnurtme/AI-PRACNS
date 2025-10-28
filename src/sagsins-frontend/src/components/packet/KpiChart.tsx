// src/components/KpiChart.tsx
import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import type { KpiData } from '../../types/type';

// Định nghĩa Props cho component này
interface KpiChartProps {
    data: KpiData[];
}

export const KpiChart: React.FC<KpiChartProps> = ( { data } ) => {
    return (
        <div className="bg-white rounded-xl shadow-lg p-4 md:p-6 transition-all duration-300 hover:shadow-2xl flex flex-col">
            <h3 className="text-center text-xl font-semibold text-slate-700 mb-4">
                Overall KPI Comparison
            </h3>
            <div className="flex-1 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={ data }>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={ { fontSize: 12 } } />
                        <YAxis tick={ { fontSize: 12 } } />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="RL" fill="#8884d8" />
                        <Bar dataKey="NonRL" fill="#82ca9d" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
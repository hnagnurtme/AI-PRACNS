// src/components/CorrelationChart.tsx
import React from 'react';
import {
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label
} from 'recharts';
// Sửa đường dẫn import (nếu file types.ts của bạn nằm ở thư mục /src)
import type { CorrelationPoint } from '../../types/type'; 
import { CustomScatterTooltip } from './CustomScatterTooltip';

// Định nghĩa Props cho component này
interface CorrelationChartProps {
    data: {
        rl: CorrelationPoint[];
        nonRl: CorrelationPoint[];
    };
}

export const CorrelationChart: React.FC<CorrelationChartProps> = ( { data } ) => {
    return (
        <div className="bg-white rounded-xl shadow-lg p-4 md:p-6 transition-all duration-300 hover:shadow-2xl flex flex-col">
            <h3 className="text-center text-xl font-semibold text-slate-700 mb-4">
                Correlation: Node Utilization vs. Hop Latency
            </h3>
            <div className="flex-1 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    {/* SỬA LỖI 1: Tăng 'bottom' margin từ 20 lên 40 để có chỗ cho legend */}
                    <ScatterChart margin={ { top: 10, right: 30, bottom: 40, left: 20 } }>
                        <CartesianGrid />
                        <XAxis type="number" dataKey="utilization" name="Utilization" unit="%" tickFormatter={ ( t: number ) => ( t * 100 ).toFixed( 0 ) } tick={ { fontSize: 12 } }>
                            <Label value="Node Utilization (%)" offset={ -15 } position="insideBottom" fill="#666" fontSize={ 14 } />
                        </XAxis>
                        <YAxis type="number" dataKey="latencyMs" name="Latency" unit="ms" tick={ { fontSize: 12 } }>
                            <Label value="Hop Latency (ms)" angle={ -90 } position="insideLeft" style={ { textAnchor: 'middle' } } fill="#666" fontSize={ 14 } />
                        </YAxis>
                        <Tooltip content={ <CustomScatterTooltip /> } />
                        {/* SỬA LỖI 2: Thêm props vào Legend để xếp chồng chúng lên nhau */}
                        <Legend 
                        
                            layout="vertical" 
                            verticalAlign="bottom" 
                            align="center" 
                        />
                        
                        <Scatter name="RL (avoids congestion)" data={ data.rl } fill="#8884d8" shape="star" />
                        <Scatter name="Non-RL (ignores congestion)" data={ data.nonRl } fill="#82ca9d" shape="cross" />
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
            <p className="text-center text-sm text-slate-500 mt-3">
                RL (stars) actively avoids busy nodes. Non-RL (crosses) is randomly distributed.
            </p>
        </div>
    );
};
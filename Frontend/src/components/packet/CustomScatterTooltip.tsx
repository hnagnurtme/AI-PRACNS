// src/components/CustomScatterTooltip.tsx
import  { type ReactNode } from 'react';
import type { CorrelationPoint } from '../../types/type';


type CustomTooltipProps = {
    active?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    payload?: any[];
};

export const CustomScatterTooltip = ( { active, payload }: CustomTooltipProps ): ReactNode => {
    if ( active && payload && payload.length ) {
        const data = payload[ 0 ].payload as CorrelationPoint;
        return (
            <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
                <p className="font-bold text-gray-700">{ payload[ 0 ].name }</p>
                <p className="text-sm text-indigo-600">
                    Utilization: { ( data.utilization * 100 ).toFixed( 1 ) }%
                </p>
                <p className="text-sm text-green-600">
                    Latency: { data.latencyMs.toFixed( 1 ) } ms
                </p>
            </div>
        );
    }
    return null;
};
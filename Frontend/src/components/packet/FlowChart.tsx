// src/components/FlowChart.tsx
import React from 'react';
import {
    Sankey, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import type { SankeyData } from '../../types/type';

// Định nghĩa Props cho component này
interface FlowChartProps {
    data: SankeyData;
    isRL: boolean;
    onShowRL: () => void;
    onShowNonRL: () => void;
}

export const FlowChart: React.FC<FlowChartProps> = ( { data, isRL, onShowRL, onShowNonRL } ) => {
    return (
        <div className="bg-white rounded-xl shadow-lg p-4 md:p-6 transition-all duration-300 hover:shadow-2xl flex flex-col">
            <h3 className="text-center text-xl font-semibold text-slate-700 mb-4">
                Packet Flow (Routing Path)
            </h3>

            {/* Nút chuyển đổi (Toggle buttons) được quản lý bởi props */ }
            <div className="flex justify-center mb-4 rounded-lg bg-slate-100 p-1">
                <button
                    onClick={ onShowRL }
                    className={ `w-1/2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${ isRL ? 'bg-indigo-600 text-white shadow' : 'text-slate-700 hover:bg-slate-200'
                        }` }
                >
                    RL Path
                </button>
                <button
                    onClick={ onShowNonRL }
                    className={ `w-1/2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${ !isRL ? 'bg-green-600 text-white shadow' : 'text-slate-700 hover:bg-slate-200'
                        }` }
                >
                    Non-RL Path
                </button>
            </div>

            <div className="flex-1 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    { data.nodes.length > 0 ? (
                        <Sankey
                            width={ 500 }
                            height={ 300 }
                            data={ data }
                            nodePadding={ 50 }
                            margin={ { top: 20, right: 20, bottom: 20, left: 20 } }
                            link={ { stroke: '#B3B3B3', strokeOpacity: 0.5 } }
                        >
                            <Tooltip />
                            { data.nodes.map( ( node ) => (
                                <Cell
                                    key={ node.name }
                                    fill={ isRL ? '#8884d8' : '#82ca9d' }
                                />
                            ) ) }
                        </Sankey>
                    ) : (
                        <div className="flex items-center justify-center h-full text-slate-500">
                            No path data to display.
                        </div>
                    ) }
                </ResponsiveContainer>
            </div>
            <p className="text-center text-sm text-slate-500 mt-3">
                Link thickness represents packet volume.
            </p>
        </div>
    );
};
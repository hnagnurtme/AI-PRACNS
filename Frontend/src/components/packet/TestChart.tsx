// src/FullDashboard.tsx
import { useState, useMemo, useCallback } from 'react';


// --- BƯỚC 1: THAY ĐỔI IMPORT ---
// Xóa toàn bộ import từ 'dataProcessor'
// Thêm import hook mới
import { usePacketProcessor } from '../../hooks/usePacketProcessor';

// Import các component biểu đồ (giữ nguyên)
import { KpiChart } from '../packet/KpiChart';
import { LatencyDistChart } from '../packet/LatencyDistChart';
import { CorrelationChart } from '../packet/CorrelationChart';
import { FlowChart } from '../packet/FlowChart';

// --- BƯỚC 2: ĐỊNH NGHĨA URL WEBSOCKET ---
// (Thay đổi URL này thành URL server của bạn)
const PACKET_STREAM_URL = import.meta.env.VITE_WS_URL; // Ví dụ: 'ws://localhost:8080/packets'

function FullDashboard () {
    // ----------------------------------------------
    // 1. STATE & DATA
    // ----------------------------------------------

    // --- BƯỚC 3: XÓA STATE CŨ ---
    // Xóa: useDemoData, realData

    // --- BƯỚC 4: GỌI HOOK MỚI ---
    // Hook này sẽ cung cấp dữ liệu biểu đồ, trạng thái kết nối, và tổng số packet
    const { chartData, isConnected, totalPackets } = usePacketProcessor( PACKET_STREAM_URL, 1000 ); 

    // State cho biểu đồ Sankey (giữ nguyên)
    const [ showRLSankey, setShowRLSankey ] = useState<boolean>( true );



    // Dữ liệu cho Sankey (giữ nguyên)
    const currentSankeyData = useMemo( () => {
        return showRLSankey ? chartData.sankeyDataRL : chartData.sankeyDataNonRL;
    }, [ showRLSankey, chartData ] );

    // Handlers cho Sankey (giữ nguyên)
    const handleShowRLSankey = useCallback( () => {
        setShowRLSankey( true );
    }, [] );

    const handleShowNonRLSankey = useCallback( () => {
        setShowRLSankey( false );
    }, [] );

    // ----------------------------------------------
    // 3. RENDER
    // ----------------------------------------------
    return (
        <div className="flex flex-col min-h-screen w-full bg-gray-100 font-sans p-4 md:p-8">

            <h1 className="text-3xl font-bold text-slate-800 text-center mb-2">
                RL vs. Non-RL Performance Dashboard
            </h1>

            {/* --- BƯỚC 7: THÊM TRẠNG THÁI KẾT NỐI --- */ }
            <div className="text-center text-sm text-slate-600 mb-6">
                <span
                    className={ `inline-block w-3 h-3 rounded-full mr-2 ${ isConnected ? 'bg-green-500' : 'bg-red-500' } 
            ${ !isConnected ? 'animate-pulse' : '' }` }
                    title={ isConnected ? 'Connected' : 'Disconnected' }
                ></span>
                { isConnected ? `Live (Total Packets: ${ totalPackets })` : 'Connecting...' }
            </div>

            {/* --- BƯỚC 8: XÓA NÚT BẤM VÀ TOAST --- */ }
            {/* Xóa: <button onClick={handleToggleData}>...</button> */ }
            {/* Xóa: <ToastContainer /> (trừ khi bạn dùng nó cho việc khác) */ }

            {/* Grid Layout (Không thay đổi) */ }
            <div className="grid grid-cols-1 md:grid-cols-2 grid-rows-4 md:grid-rows-2 gap-6 flex-1 w-full">

                {/* Các biểu đồ con không cần thay đổi gì cả.
            Chúng chỉ nhận props và render. */}

                <KpiChart data={ chartData.kpiData } />
                <LatencyDistChart data={ chartData.latencyDist } />
                <CorrelationChart data={ chartData.correlationData } />
                <FlowChart
                    data={ currentSankeyData }
                    isRL={ showRLSankey }
                    onShowRL={ handleShowRLSankey }
                    onShowNonRL={ handleShowNonRLSankey }
                />

            </div>
        </div>
    );
}

export default FullDashboard;
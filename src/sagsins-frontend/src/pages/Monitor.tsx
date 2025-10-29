import React from "react";
import { usePacketWebSocket } from "../hooks/usePacketWebSocket";
import { PacketRouteGraph } from "../components/chart/PacketRouteGraph";
import { CombinedHopMetricsChart } from "../components/chart/CombinedHopMetricsChart";

export const ComparisonDashboard: React.FC = () => {
    // Nhận realtime packets từ backend
    const packets = usePacketWebSocket( "http://localhost:8080/ws" );

    // Chỉ lấy gói cuối cùng để hiển thị
    const latest = packets.at( -1 );
    if ( !latest ) {
        return (
            <div className="p-6 text-center text-gray-500">
                Waiting for packet data...
            </div>
        );
    }

    return (
        <div className="p-6 grid grid-cols-1 gap-6">

            {/* Hàng 1: Tổng quan và Trực quan hóa Tuyến đường */ }
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">


                {/* Cột 2 & 3: Trực quan hóa Tuyến đường (Chiếm 2 cột) */ }
                <PacketRouteGraph data={ latest } />
            </div>

            {/* Hàng 2: Biểu đồ Gộp chi tiết (Luôn chiếm toàn bộ chiều rộng) */ }
            <CombinedHopMetricsChart data={ latest } />

            {/* * Các biểu đồ cũ (HopLatencyChart, HopDistanceChart, BandwidthChart) 
            * đã bị loại bỏ vì đã được gộp vào CombinedHopMetricsChart.
            */}
        </div>
    );
};

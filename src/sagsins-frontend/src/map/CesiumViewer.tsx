import React, { useEffect, useRef } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import { useNodeStore } from '../state/nodeStore';
import { syncNodeEntities, setupMapClickHandler } from './MapEntityService';
import type { NodeDTO } from '../types/NodeTypes';

interface CesiumViewerProps {
    nodes: NodeDTO[];
}

const CesiumViewer: React.FC<CesiumViewerProps> = ({ nodes }) => {
    const cesiumContainer = useRef<HTMLDivElement>(null);
    const setSelectedNode = useNodeStore((state) => state.setSelectedNode);

    // Khởi tạo Viewer lần đầu tiên
    useEffect(() => {
        if (cesiumContainer.current) {
            // Cấu hình để tải tài nguyên Cesium (CẦN THIẾT)
            // Đảm bảo bạn đã sao chép thư mục Cesium/Assets vào public/Cesium
            // (Thường được xử lý bởi cấu hình Vite/Webpack)
            // Cesium.buildModuleUrl.setBaseUrl('/Cesium/'); 

            const viewer = new Cesium.Viewer(cesiumContainer.current, {
                // Tắt các tính năng không cần thiết
                timeline: false,
                animation: false,
                baseLayerPicker: false,
                geocoder: false,
                homeButton: false,
                sceneModePicker: false,
                navigationHelpButton: false,
                infoBox: false,
            });

            // Thiết lập handler click
            const handler = setupMapClickHandler(viewer, setSelectedNode);
            
            // Cleanup khi component bị unmount
            return () => {
                handler.destroy();
                viewer.destroy();
            };
        }
    }, [setSelectedNode]);

    // Đồng bộ hóa entities mỗi khi danh sách nodes thay đổi
    useEffect(() => {
        if (Cesium.defined(window.viewer)) {
            syncNodeEntities(window.viewer, nodes); 
        }
    }, [nodes]);

    return <div ref={cesiumContainer} className="w-full h-screen" />;
};

export default CesiumViewer;
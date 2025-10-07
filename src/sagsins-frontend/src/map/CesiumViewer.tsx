// src/map/CesiumViewer.tsx

import React, { useEffect, useRef } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import { useNodeStore } from '../state/nodeStore';
import { syncNodeEntities, setupMapClickHandler } from './MapEntityService';
import type { NodeDTO } from '../types/NodeTypes';

// Khai báo lại window interface đã được sửa chữa trong global.d.ts
declare global {
    interface Window {
        viewer?: Cesium.Viewer;
    }
}

interface CesiumViewerProps {
    nodes: NodeDTO[];
}

const CesiumViewer: React.FC<CesiumViewerProps> = ({ nodes }) => {
    const cesiumContainer = useRef<HTMLDivElement>(null);
    const setSelectedNode = useNodeStore((state) => state.setSelectedNode);

    // 1. Khởi tạo Viewer lần đầu tiên (Chỉ chạy một lần)
    useEffect(() => {
        if (cesiumContainer.current) {
            
            const viewer = new Cesium.Viewer(cesiumContainer.current, {
                timeline: true,
                animation: true,
                baseLayerPicker: false,
                geocoder: true,
                homeButton: true,
                sceneModePicker: true,
                navigationHelpButton: false,
                infoBox: false,
            }
        );
            


            window.viewer = viewer; 

            // Thiết lập handler click
            const handler = setupMapClickHandler(viewer, setSelectedNode);
            
            // Cleanup khi component bị unmount
            return () => {
                handler.destroy();
                viewer.destroy();
                window.viewer = undefined; // Dọn dẹp
            };
        }
    }, [setSelectedNode]);


    // 2. Đồng bộ hóa entities và điều chỉnh camera mỗi khi danh sách nodes thay đổi
    useEffect(() => {
        const viewer = window.viewer;

        if (Cesium.defined(viewer)) {
            syncNodeEntities(viewer, nodes); 
            
            // THÊM LOGIC FLY TO để đảm bảo bạn thấy các Node
            if (nodes.length > 0) {
                    const firstNode = nodes[0];
                    
                    viewer.camera.flyTo({
                        destination: Cesium.Cartesian3.fromDegrees(
                            firstNode.position.longitude, 
                            firstNode.position.latitude, 
                            // Độ cao của Node + một offset lớn (5000 km) để nhìn thấy
                            firstNode.position.altitude * 1000 + 5000000 
                        ),
                        duration: 2.0 
                    });
            }
        }
    }, [nodes]);

    return <div ref={cesiumContainer} className="w-full h-screen" />;
};

export default CesiumViewer;
// src/map/MapEntityService.ts

import * as Cesium from 'cesium';
import type { NodeDTO } from '../types/NodeTypes';

// --- Hằng số ---
const PATH_COLOR = Cesium.Color.YELLOW.withAlpha(0.7);
const ORBIT_DURATION_HOURS = 2; 
const SAFE_BASE_DRIFT_DEG = 0.1; 

// --- Icon SVG paths ---
const SATELLITE_ICON = '/icons/SATELLITE.svg';
const GROUND_STATION_ICON = '/icons/STATION.svg';



const createStaticPosition = (node: NodeDTO): Cesium.Cartesian3 => {
    return Cesium.Cartesian3.fromDegrees(
        node.position.longitude,
        node.position.latitude,
        node.position.altitude * 1000 // Chuyển Km sang Mét
    );
};

interface SampledPositionPropertyWithAvailability extends Cesium.SampledPositionProperty {
    setAvailability(availability: Cesium.TimeIntervalCollection): void;
}


/**
 * TẠO VỊ TRÍ ĐỘNG (SampledPositionProperty) cho Vệ tinh.
 * LOGIC GIẢ ĐỊNH: Tạo 120 điểm trong vòng 2 giờ.
 */
const createDynamicPosition = (node: NodeDTO): Cesium.PositionProperty => {
    const currentTime = Cesium.JulianDate.now(); 
    
    // Đặt thời gian bắt đầu lấy mẫu là THỜI GIAN HIỆN TẠI
    const startTime = currentTime;
    const totalDurationSeconds = 3600 * ORBIT_DURATION_HOURS; 
    const sampleRateSeconds = 60;
    const numSamples = Math.floor(totalDurationSeconds / sampleRateSeconds);

    const sampledPosition = new Cesium.SampledPositionProperty();
    (sampledPosition as unknown as SampledPositionPropertyWithAvailability).setInterpolationOptions({
        interpolationDegree: 5,
        interpolationAlgorithm: Cesium.LagrangePolynomialApproximation
    });

    sampledPosition.setInterpolationOptions({
        interpolationDegree: 5,
        interpolationAlgorithm: Cesium.LagrangePolynomialApproximation
    });

    
    const initialLon = node.position.longitude;
    const initialLat = node.position.latitude;
    const initialAltMeters = node.position.altitude * 1000;

    const lonDriftRatePerSecond = SAFE_BASE_DRIFT_DEG / totalDurationSeconds; 
    const latDriftRatePerSecond = SAFE_BASE_DRIFT_DEG / totalDurationSeconds; 




    for (let i = 0; i <= numSamples; i++) {
        const timeOffsetSeconds = i * sampleRateSeconds;
        const time = Cesium.JulianDate.addSeconds(startTime, timeOffsetSeconds, new Cesium.JulianDate());

        // *************** LOGIC MÔ PHỎNG VỊ TRÍ ***************
        // Calculate the new positions using the verified numeric drift rates
        const newLon = initialLon + lonDriftRatePerSecond * timeOffsetSeconds;
        const newLat = initialLat + latDriftRatePerSecond * timeOffsetSeconds;
        
        const position = Cesium.Cartesian3.fromDegrees(
            newLon, 
            newLat, 
            initialAltMeters
        );
        // ******************************************************

        sampledPosition.addSample(time, position);
    }


    return sampledPosition;
};


/**
 * Thêm hoặc cập nhật các entities Node và Path.
 */
export const syncNodeEntities = (viewer: Cesium.Viewer, nodes: NodeDTO[]): void => {
    viewer.entities.removeAll(); 

    nodes.forEach(node => {
        const isSatellite = node.nodeType.includes('SATELLITE');
        const iconPath = isSatellite ? SATELLITE_ICON : GROUND_STATION_ICON;
        
        const isDynamic = isSatellite && node.orbit && node.velocity;
        
        // Xác định vị trí
        const entityPosition = isDynamic 
            ? createDynamicPosition(node) 
            : createStaticPosition(node) as unknown as Cesium.PositionProperty; 


        // 1. Entity chính (Sử dụng billboard với SVG icon)
        viewer.entities.add({
            id: node.nodeId,
            position: entityPosition, 
            name: node.nodeId,
            
            billboard: {
                image: iconPath,
                width: isSatellite ? 32 : 40,
                height: isSatellite ? 32 : 40,
                verticalOrigin: Cesium.VerticalOrigin.CENTER,
                horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
                heightReference: Cesium.HeightReference.NONE,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
                // Tự động scale theo khoảng cách camera
                scaleByDistance: new Cesium.NearFarScalar(
                    1.5e6,  // Khoảng cách gần (1.5 triệu mét)
                    1.2,    // Scale khi gần
                    2.0e7,  // Khoảng cách xa (20 triệu mét)
                    0.3     // Scale khi xa
                ),
                // Độ trong suốt theo khoảng cách
                translucencyByDistance: new Cesium.NearFarScalar(
                    1.5e6,  // Khoảng cách gần
                    1.0,    // Hoàn toàn rõ khi gần
                    2.5e7,  // Khoảng cách xa
                    0.4     // Mờ đi khi xa
                ),
                // Điều chỉnh vị trí nếu cần
                pixelOffset: new Cesium.Cartesian2(0, 0),
            },
            
            properties: { nodeData: node as unknown } 
        });

        
        // 2. Entity Quỹ đạo (Chỉ cho Vệ tinh)
        if (isDynamic) {
             viewer.entities.add({
                id: `${node.nodeId}_path`,
                position: entityPosition,
                name: `${node.nodeId} Orbit Path`,
                path: {
                    leadTime: ORBIT_DURATION_HOURS * 3600 * 0.5, 
                    trailTime: 0, 
                    width: 2,
                    material: PATH_COLOR,
                },
            });
        }
    });
};


/**
 * Cấu hình handler xử lý sự kiện click trên bản đồ.
 */
export const setupMapClickHandler = (viewer: Cesium.Viewer, setSelectedNode: (node: NodeDTO | null) => void): Cesium.ScreenSpaceEventHandler => {
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    handler.setInputAction((click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const picked = viewer.scene.pick(click.position);
        
        if (Cesium.defined(picked) && Cesium.defined(picked.id) && picked.id.properties && picked.id.properties.nodeData) {
            const nodeProperty = picked.id.properties.nodeData.getValue();
            const nodeData: NodeDTO = nodeProperty as NodeDTO; 
            setSelectedNode(nodeData);
        } else {
            setSelectedNode(null); 
        }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    
    return handler;
};
// src/map/CesiumViewer.tsx
import React, { useEffect, useRef, useCallback } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { useNodeStore } from "../state/nodeStore"; // Đảm bảo import store đã cập nhật
import type { NodeDTO } from "../types/NodeTypes";

// Icon SVG import
import SATELLITEICON from "../assets/icons/SATELLITE.svg";
import STATIONICON from "../assets/icons/STATION.svg";

declare global {
    interface Window {
        viewer?: Cesium.Viewer;
    }
}

interface CesiumViewerProps {
    nodes: NodeDTO[];
}

// ================== Constants ==================
const ORBIT_CONFIG = {
    LEO: { altitude: 550000, period: 5400 },
    MEO: { altitude: 20000000, period: 43200 },
    GEO: { altitude: 35786000, period: 86400 },
};

const VISUAL_CONFIG = {
    SATELLITE: {
        size: 24,
        scale: { near: 1.8, mid: 1.0, far: 0.1 },
    },
    STATION: {
        size: 20,
        scale: { near: 1.5, mid: 1.0, far: 0.1 },
    },
};

// ================== Utility ==================
const getNodeColor = (nodeType: string): Cesium.Color => {
    switch (nodeType) {
        case "LEO_SATELLITE": return Cesium.Color.CYAN;
        case "MEO_SATELLITE": return Cesium.Color.GOLD;
        case "GEO_SATELLITE": return Cesium.Color.ORANGE;
        case "GROUND_STATION": return Cesium.Color.LIME;
        default: return Cesium.Color.WHITE;
    }
};

const getOrbitConfig = (nodeType: string) => {
    switch (nodeType) {
        case "LEO_SATELLITE": return ORBIT_CONFIG.LEO;
        case "MEO_SATELLITE": return ORBIT_CONFIG.MEO;
        case "GEO_SATELLITE": return ORBIT_CONFIG.GEO;
        default: return ORBIT_CONFIG.LEO;
    }
};

// ================== Tạo chuyển động tròn ==================
const createCircularOrbit = (
    node: NodeDTO,
    orbitConfig: { altitude: number; period: number }
): Cesium.SampledPositionProperty => {
    const positionProperty = new Cesium.SampledPositionProperty();
    const lon = Number(node.position?.longitude) || 0;
    const lat = Number(node.position?.latitude) || 0;
    const epoch = Cesium.JulianDate.now();
    const totalSamples = 720;
    const totalDuration = orbitConfig.period * 2;
    const step = totalDuration / totalSamples;
    const angularSpeed = (2 * Math.PI) / orbitConfig.period;

    for (let i = 0; i <= totalSamples; i++) {
        const timeOffset = i * step;
        const time = Cesium.JulianDate.addSeconds(epoch, timeOffset, new Cesium.JulianDate());
        const angle_rad = angularSpeed * timeOffset;
        const newLon = (lon + (angle_rad * 180 / Math.PI)) % 360;
        const position = Cesium.Cartesian3.fromDegrees(newLon, lat, orbitConfig.altitude);
        positionProperty.addSample(time, position);
    }
    positionProperty.setInterpolationOptions({
        interpolationDegree: 5,
        interpolationAlgorithm: Cesium.LagrangePolynomialApproximation,
    });
    return positionProperty;
};

// ================== Component ==================
const CesiumViewer: React.FC<CesiumViewerProps> = ({ nodes }) => {
    const cesiumContainer = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<Cesium.Viewer | null>(null);
    const entityCacheRef = useRef<Map<string, Cesium.Entity>>(new Map());

    // [SỬA] Lấy 'flyToTrigger' từ store
    const { setSelectedNode, selectedNode, cameraFollowMode, flyToTrigger , setCameraFollowMode } = useNodeStore();
    
    // Thêm ref để theo dõi giá trị trigger (tránh lặp vô hạn)
    const lastTriggerRef = useRef(flyToTrigger);

    // ========== Tạo Billboard ==========
    const createBillboardOptions = useCallback((node: NodeDTO, isSatellite: boolean) => {
        const config = isSatellite ? VISUAL_CONFIG.SATELLITE : VISUAL_CONFIG.STATION;
        const color = getNodeColor(node.nodeType);
        return {
            image: isSatellite ? SATELLITEICON : STATIONICON,
            width: config.size,
            height: config.size,
            color,
            scaleByDistance: new Cesium.NearFarScalar(5e4, config.scale.near, 1e8, config.scale.far),
            pixelOffsetScaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.3),
            translucencyByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.4),
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 2e8),
            heightReference: isSatellite
                ? Cesium.HeightReference.NONE
                : Cesium.HeightReference.CLAMP_TO_GROUND,
        };
    }, []);

    // ========== Tạo Label ==========
    const createLabelOptions = useCallback((node: NodeDTO) => {
        const color = getNodeColor(node.nodeType);
        const shortName = node.nodeName.length > 8 
            ? node.nodeName.substring(0, 6) + "…" 
            : node.nodeName;
        return new Cesium.LabelGraphics({
            text: shortName,
            font: 'bold 14px sans-serif',
            fillColor: color,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            pixelOffset: new Cesium.Cartesian2(0, -25),
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.2, 5e6, 0),
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e6),
        });
    }, []);

    // ========== Thêm Node ==========
    const addNode = useCallback((node: NodeDTO) => {
        if (!viewerRef.current) return;
        const { nodeId, nodeType, position } = node;
        const lon = Number(position?.longitude);
        const lat = Number(position?.latitude);
        if (isNaN(lon) || isNaN(lat)) return;

        const isSatellite = nodeType.includes("SATELLITE");
        let positionProperty: Cesium.PositionProperty;

        if (isSatellite) {
            const orbitConfig = getOrbitConfig(nodeType);
            positionProperty = createCircularOrbit(node, orbitConfig);
        } else {
            const staticPos = Cesium.Cartesian3.fromDegrees(lon, lat, 100);
            positionProperty = new Cesium.ConstantPositionProperty(staticPos);
        }

        const existing = entityCacheRef.current.get(nodeId);
        if (existing) {
            viewerRef.current.entities.remove(existing);
        }

        const entity = viewerRef.current.entities.add({
            id: nodeId,
            name: node.nodeName,
            position: positionProperty,
            billboard: createBillboardOptions(node, isSatellite),
            label: createLabelOptions(node),
        });
        entityCacheRef.current.set(nodeId, entity);
    }, [createBillboardOptions, createLabelOptions]);

    // ========== Batch Add ==========
    const batchAddNodes = useCallback((nodesToAdd: NodeDTO[]) => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        viewer.entities.suspendEvents();
        entityCacheRef.current.clear();
        nodesToAdd.forEach(addNode);
        viewer.entities.resumeEvents();
    }, [addNode]);

    // ========== Initialize Cesium (Chỉ chạy 1 lần) ==========
    useEffect(() => {
        if (cesiumContainer.current && !viewerRef.current) {
            const viewer = new Cesium.Viewer(cesiumContainer.current, {
                timeline: true,
                animation: true,
                baseLayerPicker: false,
                geocoder: false,
                homeButton: true,
                sceneModePicker: true,
                navigationHelpButton: false,
                infoBox: false,
                selectionIndicator: false,
            });
            viewer.scene.requestRenderMode = true;
            viewer.scene.globe.depthTestAgainstTerrain = true;
            viewer.scene.globe.enableLighting = false;
            viewer.scene.fog.enabled = false;
            viewer.scene.globe.baseColor = Cesium.Color.DARKSLATEGRAY;
            viewer.scene.backgroundColor = Cesium.Color.BLACK;
            viewer.camera.setView({
                destination: Cesium.Cartesian3.fromDegrees(0, 0, 2e7),
            });
            viewer.clock.shouldAnimate = true;
            viewer.clock.multiplier = 120;
            viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;

            viewerRef.current = viewer;
            window.viewer = viewer;
            
            return () => {
                viewer.destroy();
                viewerRef.current = null;
                window.viewer = undefined;
            };
        }
    }, []);

    // ========== useEffect cho Click Handler (Click map CHỈ HIỆN DETAIL) ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        
        const clickHandler = (event: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
            const picked = viewer.scene.pick(event.position);
            
            if (picked?.id) {
                const node = nodes.find(n => n.nodeId === picked.id.id);
                setSelectedNode(node || null); // Chỉ gọi setSelectedNode
            } else {
                setSelectedNode(null);
            }
        };

        viewer.screenSpaceEventHandler.setInputAction(
            clickHandler,
            Cesium.ScreenSpaceEventType.LEFT_CLICK
        );

        return () => {
            const currentViewer = viewerRef.current;
            if (currentViewer && !currentViewer.isDestroyed()) {
                currentViewer.screenSpaceEventHandler.removeInputAction(
                    Cesium.ScreenSpaceEventType.LEFT_CLICK
                );
            }
        };
    }, [nodes, setSelectedNode]); // Phụ thuộc vào `nodes`


    // ========== Update Nodes & Clock ==========
    useEffect(() => {
        if (!viewerRef.current || !nodes.length) return;
        const viewer = viewerRef.current;
        
        viewer.entities.removeAll();
        batchAddNodes(nodes);

        const startTime = Cesium.JulianDate.now();
        const stopTime = Cesium.JulianDate.addSeconds(startTime, ORBIT_CONFIG.LEO.period * 2, new Cesium.JulianDate());
        
        viewer.clock.startTime = startTime;
        viewer.clock.stopTime = stopTime;
        viewer.clock.currentTime = startTime;
        viewer.timeline?.zoomTo(startTime, stopTime);

        const firstSat = nodes.find(n => n.nodeType.includes("SATELLITE"));
        if (firstSat) {
            setTimeout(() => {
                const entity = viewer.entities.getById(firstSat.nodeId);
                if (entity) {
                    viewer.flyTo(entity, { duration: 2 });
                }
            }, 500);
        }
    }, [nodes, batchAddNodes]);

    // ========== Camera Follow ==========
    useEffect(() => {
        if (!viewerRef.current) return;
        const viewer = viewerRef.current;
        let followListener: Cesium.Event.RemoveCallback | undefined;

        if (selectedNode && cameraFollowMode) {
            const entity = viewer.entities.getById(selectedNode.nodeId);
            if (!entity) return;

            const follow = (clock: Cesium.Clock) => {
                const pos = entity.position?.getValue(clock.currentTime);
                if (pos) {
                    viewer.camera.lookAt(pos, new Cesium.HeadingPitchRange(0, -Cesium.Math.PI_OVER_THREE, 2e5));
                }
            };
            followListener = viewer.clock.onTick.addEventListener(follow);
        }

        return () => {
            if (followListener) {
                followListener();
            }
        };
    }, [selectedNode, cameraFollowMode]);

    // Effect riêng để reset camera khi TẮT follow mode
    useEffect(() => {
        if (viewerRef.current && !cameraFollowMode) {
            viewerRef.current.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
        }
    }, [cameraFollowMode]);

    // ========== [MỚI] useEffect để XỬ LÝ BAY (từ Sidebar) ==========
    useEffect(() => {
        // Chỉ chạy nếu trigger THỰC SỰ thay đổi
        if (flyToTrigger === lastTriggerRef.current) {
            return;
        }

        // Cập nhật ref
        lastTriggerRef.current = flyToTrigger;
        
        if (!viewerRef.current || !selectedNode) {
            return;
        }

        const viewer = viewerRef.current;
        const entity = viewer.entities.getById(selectedNode.nodeId);
        
        if (entity) {
            const isSatellite = selectedNode.nodeType.includes("SATELLITE");
            
            viewer.flyTo(entity, {
                duration: 1.5,
                offset: new Cesium.HeadingPitchRange(
                    0,
                    -Cesium.Math.PI_OVER_THREE,
                    isSatellite ? 800000 : 100000 
                ),
            }).then((finished) => {
    // [SỬA] Callback .then() sẽ chạy khi bay xong
    // 'finished' là true nếu camera bay đến nơi
    if (finished) {
        setCameraFollowMode(true);
    }
});
        }
    }, [flyToTrigger, selectedNode, cameraFollowMode,setCameraFollowMode ]); 

    return <div ref={cesiumContainer} className="w-full h-screen" />;
};

export default CesiumViewer;
// src/map/CesiumViewer.tsx

import React, { useEffect, useRef, useCallback } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import { useNodeStore } from '../state/nodeStore';
import type { NodeDTO } from '../types/NodeTypes';

// Khai báo lại window interface đã được sửa chữa trong global.d.ts
declare global {
    interface Window {
        viewer?: Cesium.Viewer;
    }
}

// ====================== Orbital Mechanics Functions ======================
const calculateOrbitalPeriod = (semiMajorAxisKm: number): number => {
    // Earth's gravitational parameter (GM) in km³/s²
    const GM = 398600.4418;
    // Calculate orbital period using Kepler's third law
    const period = 2 * Math.PI * Math.sqrt(Math.pow(semiMajorAxisKm, 3) / GM);
    return period; // seconds
};

const calculateOrbitalPosition = (orbit: NonNullable<NodeDTO['orbit']>, trueAnomalyDeg: number): Cesium.Cartesian3 => {
    // Convert degrees to radians
    const trueAnomaly = Cesium.Math.toRadians(trueAnomalyDeg);
    const inclination = Cesium.Math.toRadians(orbit.inclinationDeg);
    const raan = Cesium.Math.toRadians(orbit.raanDeg);
    const argOfPeriapsis = Cesium.Math.toRadians(orbit.argumentOfPerigeeDeg);
    
    // Calculate distance from focus
    const r = orbit.semiMajorAxisKm * (1 - orbit.eccentricity * orbit.eccentricity) / 
             (1 + orbit.eccentricity * Math.cos(trueAnomaly));
    
    // Position in orbital plane
    const xOrbital = r * Math.cos(trueAnomaly);
    const yOrbital = r * Math.sin(trueAnomaly);
    
    // Rotate to Earth-centered coordinates
    const cosRaan = Math.cos(raan);
    const sinRaan = Math.sin(raan);
    const cosInc = Math.cos(inclination);
    const sinInc = Math.sin(inclination);
    const cosArg = Math.cos(argOfPeriapsis);
    const sinArg = Math.sin(argOfPeriapsis);
    
    const x = (cosRaan * cosArg - sinRaan * sinArg * cosInc) * xOrbital + 
              (-cosRaan * sinArg - sinRaan * cosArg * cosInc) * yOrbital;
    const y = (sinRaan * cosArg + cosRaan * sinArg * cosInc) * xOrbital + 
              (-sinRaan * sinArg + cosRaan * cosArg * cosInc) * yOrbital;
    const z = (sinInc * sinArg) * xOrbital + (sinInc * cosArg) * yOrbital;
    
    // Convert from km to meters and create Cartesian3
    return new Cesium.Cartesian3(x * 1000, y * 1000, z * 1000);
};

// ====================== Utils ======================
function getNodeColor(nodeType: string): Cesium.Color {
    switch (nodeType) {
        case "LEO_SATELLITE":
        case "MEO_SATELLITE":
        case "GEO_SATELLITE":
            return Cesium.Color.GOLD;
        case "GROUND_STATION":
            return Cesium.Color.CYAN;
        default:
            return Cesium.Color.WHITE;
    }
}

// ---- ICONS (SVG) ----
function createSatelliteIcon(color: string): string {
    return `
    <svg width="32" height="32" xmlns="http://www.w3.org/2000/svg">
      <circle cx="16" cy="16" r="6" fill="${color}" stroke="#000" stroke-width="1"/>
      <rect x="13" y="8" width="6" height="2" fill="${color}" opacity="0.8"/>
      <rect x="13" y="22" width="6" height="2" fill="${color}" opacity="0.8"/>
      <rect x="8" y="13" width="2" height="6" fill="${color}" opacity="0.8"/>
      <rect x="22" y="13" width="2" height="6" fill="${color}" opacity="0.8"/>
    </svg>`;
}

function createGroundStationIcon(color: string): string {
    return `
    <svg width="28" height="28" xmlns="http://www.w3.org/2000/svg">
      <polygon points="14,4 18,20 10,20" fill="${color}" stroke="black" stroke-width="1"/>
      <line x1="14" y1="4" x2="14" y2="24" stroke="${color}" stroke-width="2"/>
      <circle cx="14" cy="24" r="3" fill="${color}" />
    </svg>`;
}

interface CesiumViewerProps {
    nodes: NodeDTO[];
}

const CesiumViewer: React.FC<CesiumViewerProps> = ({ nodes }) => {
    const cesiumContainer = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<Cesium.Viewer | null>(null);
    const { setSelectedNode, selectedNode, cameraFollowMode } = useNodeStore();
    const animationFrameRef = useRef<number | null>(null);

    // ====================== Node Management ======================
    const addNode = useCallback((node: NodeDTO) => {
        if (!viewerRef.current) return;

        const viewer = viewerRef.current;
        const { nodeId, nodeType, position, orbit, velocity } = node;
        const lon = Number(position?.longitude);
        const lat = Number(position?.latitude);
        const alt = Number(position?.altitude) * 1000 || 550000; // Convert km to meters
        const color = getNodeColor(nodeType);

        if (isNaN(lon) || isNaN(lat) || lon < -180 || lon > 180 || lat < -90 || lat > 90) {
            console.warn("❌ Invalid node data:", node);
            return;
        }

        const isSelected = selectedNode?.nodeId === nodeId;
        const isSatellite = nodeType.includes('SATELLITE');

        // Create SVG icon
        let iconSvg = "";
        if (isSatellite) {
            iconSvg = createSatelliteIcon(color.toCssColorString());
        } else {
            iconSvg = createGroundStationIcon(color.toCssColorString());
        }

        // Create position with animation for satellites
        let entityPosition;
        if (isSatellite && (orbit || velocity)) {
            // Create realistic orbital motion using actual orbit/velocity data
            const currentTime = Cesium.JulianDate.now();
            const sampledPosition = new Cesium.SampledPositionProperty();
            
            sampledPosition.setInterpolationOptions({
                interpolationDegree: 5,
                interpolationAlgorithm: Cesium.LagrangePolynomialApproximation
            });

            if (orbit) {
                // Use real orbital parameters for motion
                const orbitalPeriod = calculateOrbitalPeriod(orbit.semiMajorAxisKm);
                const numSamples = 360; // More samples for smoother motion
                
                for (let i = 0; i <= numSamples; i++) {
                    const time = Cesium.JulianDate.addSeconds(currentTime, (i / numSamples) * orbitalPeriod, new Cesium.JulianDate());
                    
                    // Calculate position using orbital mechanics
                    const trueAnomaly = orbit.trueAnomalyDeg + (360 * i / numSamples);
                    const orbitalPosition = calculateOrbitalPosition(orbit, trueAnomaly);
                    
                    sampledPosition.addSample(time, orbitalPosition);
                }
            } else if (velocity) {
                // Use velocity vector for linear motion simulation
                const totalTime = 7200; // 2 hours
                const numSamples = 120;
                
                for (let i = 0; i <= numSamples; i++) {
                    const time = Cesium.JulianDate.addSeconds(currentTime, (i / numSamples) * totalTime, new Cesium.JulianDate());
                    const deltaTime = (i / numSamples) * totalTime;
                    
                    // Simple velocity integration
                    const newLon = lon + (velocity.velocityX * deltaTime) / 111.32; // Rough conversion km to degrees
                    const newLat = lat + (velocity.velocityY * deltaTime) / 110.54;
                    const newAlt = alt + (velocity.velocityZ * deltaTime * 1000); // km to meters
                    
                    const positionCartesian = Cesium.Cartesian3.fromDegrees(newLon, newLat, newAlt);
                    sampledPosition.addSample(time, positionCartesian);
                }
            }
            
            entityPosition = sampledPosition;
        } else {
            // Static position for ground stations or satellites without orbit data
            entityPosition = Cesium.Cartesian3.fromDegrees(lon, lat, alt);
        }

        const entityOptions: Cesium.Entity.ConstructorOptions = {
            id: nodeId,
            name: `${nodeType}: ${nodeId}`,
            position: entityPosition,
            billboard: {
                image: "data:image/svg+xml;base64," + btoa(iconSvg),
                scale: isSelected ? 1.3 : 1.0,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
                scaleByDistance: new Cesium.NearFarScalar(1.5e6, 1.2, 2.0e7, 0.5),
                translucencyByDistance: new Cesium.NearFarScalar(1.5e6, 1.0, 2.5e7, 0.6),
                heightReference: Cesium.HeightReference.NONE,
            },
            label: {
                text: nodeId.substring(0, 8) + "...",
                font: "12px monospace",
                fillColor: color,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                verticalOrigin: Cesium.VerticalOrigin.TOP,
                pixelOffset: new Cesium.Cartesian2(0, -20),
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
                distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 8_000_000),
            },
        };

        return viewer.entities.add(entityOptions);
    }, [selectedNode]);

    // ====================== Initialization ======================
    useEffect(() => {
        if (cesiumContainer.current && !viewerRef.current) {
            const viewer = new Cesium.Viewer(cesiumContainer.current, {
                timeline: true,
                animation: true,
                baseLayerPicker: false,
                geocoder: true,
                homeButton: true,
                sceneModePicker: true,
                navigationHelpButton: false,
                infoBox: false,
            });

            viewerRef.current = viewer;
            window.viewer = viewer;

            // Set initial camera position
            viewer.camera.setView({
                destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000),
                orientation: {
                    heading: 0,
                    pitch: -Cesium.Math.PI_OVER_TWO,
                    roll: 0,
                },
            });

            // Enable animation
            viewer.clock.shouldAnimate = true;
            viewer.clock.multiplier = 1.0;

            // Click handler
            viewer.cesiumWidget.screenSpaceEventHandler.setInputAction(
                (event: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
                    const pickedObject = viewer.scene.pick(event.position);
                    if (pickedObject && pickedObject.id) {
                        const nodeId = pickedObject.id.id;
                        const node = nodes.find(n => n.nodeId === nodeId);
                        setSelectedNode(node || null);
                    } else {
                        setSelectedNode(null);
                    }
                },
                Cesium.ScreenSpaceEventType.LEFT_CLICK
            );

            return () => {
                if (viewer && !viewer.isDestroyed()) {
                    viewer.destroy();
                }
                viewerRef.current = null;
                window.viewer = undefined;
            };
        }
    }, [nodes, setSelectedNode]);

    // ====================== Update Entities ======================
    useEffect(() => {
        if (!viewerRef.current) return;
        
        const viewer = viewerRef.current;
        viewer.entities.removeAll();
        nodes.forEach((node) => addNode(node));

        // Fly to first node if available
        if (nodes.length > 0) {
            const firstNode = nodes[0];
            viewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(
                    firstNode.position.longitude,
                    firstNode.position.latitude,
                    firstNode.position.altitude * 1000 + 5000000
                ),
                duration: 2.0
            });
        }
    }, [nodes, addNode]);

    // ====================== Camera Follow Logic ======================
    useEffect(() => {
        if (!viewerRef.current || !selectedNode || !cameraFollowMode) {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            return;
        }

        const viewer = viewerRef.current;
        const entity = viewer.entities.getById(selectedNode.nodeId);
        
        if (!entity || !entity.position) {
            return;
        }

        const followCamera = () => {
            if (!viewer || viewer.isDestroyed() || !cameraFollowMode || !entity.position) {
                return;
            }

            const currentTime = viewer.clock.currentTime;
            const position = entity.position.getValue(currentTime);
            
            if (position) {
                // Calculate offset position for better viewing
                const cartographic = Cesium.Cartographic.fromCartesian(position);
                const offsetDistance = 100000; // 100km offset for better perspective
                
                // Create a smooth camera position
                const cameraPosition = Cesium.Cartesian3.fromRadians(
                    cartographic.longitude + 0.01, // Small longitude offset
                    cartographic.latitude + 0.005, // Small latitude offset  
                    cartographic.height + offsetDistance
                );

                // Smooth camera movement
                viewer.camera.setView({
                    destination: cameraPosition,
                    orientation: {
                        heading: 0,
                        pitch: -0.7, // Look down at an angle
                        roll: 0,
                    },
                });
            }

            animationFrameRef.current = requestAnimationFrame(followCamera);
        };

        followCamera();

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, [selectedNode, cameraFollowMode]);

    // ====================== Handle Selection ======================
    useEffect(() => {
        if (!viewerRef.current || !selectedNode || cameraFollowMode) return;
        
        const viewer = viewerRef.current;
        const entity = viewer.entities.getById(selectedNode.nodeId);
        if (entity) {
            viewer.flyTo(entity, {
                duration: 2.0,
                offset: new Cesium.HeadingPitchRange(0, -0.5, 1000000),
            });
        }
    }, [selectedNode, cameraFollowMode]);

    return <div ref={cesiumContainer} className="w-full h-screen" />;
};

export default CesiumViewer;
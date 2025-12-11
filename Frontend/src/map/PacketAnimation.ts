import * as Cesium from 'cesium';
import type { RoutingPath } from '../types/RoutingTypes';

export interface PacketAnimation {
    packetId: string;
    path: RoutingPath;
    entity: Cesium.Entity;
    startTime: Cesium.JulianDate;
    endTime: Cesium.JulianDate;
    duration: number; // seconds
}

/**
 * Create animated packet entity that moves along the path
 */
export const createPacketAnimation = (
    viewer: Cesium.Viewer,
    packetId: string,
    path: RoutingPath,
    duration: number = 5 // seconds
): PacketAnimation | null => {
    if (!path || path.path.length < 2) {
        return null;
    }

    // Use current clock time from viewer, but ensure it's within the clock's time range
    const clock = viewer.clock;
    let startTime: Cesium.JulianDate;
    
    // If clock is not animating or time is not set, use now
    if (!clock.shouldAnimate || !clock.currentTime) {
        startTime = Cesium.JulianDate.now();
    } else {
        startTime = clock.currentTime.clone();
    }
    
    // Ensure startTime is within clock's time range
    if (clock.startTime && Cesium.JulianDate.lessThan(startTime, clock.startTime)) {
        startTime = clock.startTime.clone();
    }
    if (clock.stopTime && Cesium.JulianDate.greaterThan(startTime, clock.stopTime)) {
        startTime = clock.startTime.clone();
    }
    
    const endTime = Cesium.JulianDate.addSeconds(startTime, duration, new Cesium.JulianDate());
    
    // Extend clock time range if needed
    if (!clock.stopTime || Cesium.JulianDate.greaterThan(endTime, clock.stopTime)) {
        clock.stopTime = Cesium.JulianDate.addSeconds(endTime, 1, new Cesium.JulianDate());
    }

    // Create position property that moves along the path
    const positionProperty = new Cesium.SampledPositionProperty();
    
    // Sample positions along the path
    const numSamples = Math.max(100, path.path.length * 20); // More samples for smoother animation
    for (let i = 0; i <= numSamples; i++) {
        const t = i / numSamples;
        const segmentIndex = Math.floor(t * (path.path.length - 1));
        const segmentT = (t * (path.path.length - 1)) - segmentIndex;
        
        const startSegment = path.path[segmentIndex];
        const endSegment = path.path[Math.min(segmentIndex + 1, path.path.length - 1)];
        
        if (!startSegment || !endSegment) continue;
        
        const startPos = Cesium.Cartesian3.fromDegrees(
            startSegment.position.longitude,
            startSegment.position.latitude,
            startSegment.position.altitude || 0
        );
        const endPos = Cesium.Cartesian3.fromDegrees(
            endSegment.position.longitude,
            endSegment.position.latitude,
            endSegment.position.altitude || 0
        );
        
        const currentPos = Cesium.Cartesian3.lerp(startPos, endPos, segmentT, new Cesium.Cartesian3());
        const time = Cesium.JulianDate.addSeconds(startTime, t * duration, new Cesium.JulianDate());
        positionProperty.addSample(time, currentPos);
    }
    
    positionProperty.setInterpolationOptions({
        interpolationDegree: 1,
        interpolationAlgorithm: Cesium.LinearApproximation,
    });

    // Create packet entity (small glowing sphere)
    const entity = viewer.entities.add({
        id: `PACKET-${packetId}`,
        position: positionProperty,
        availability: new Cesium.TimeIntervalCollection([
            new Cesium.TimeInterval({
                start: startTime,
                stop: endTime,
            }),
        ]),
        point: {
            pixelSize: 12,
            color: Cesium.Color.YELLOW,
            outlineColor: Cesium.Color.ORANGE,
            outlineWidth: 3,
            heightReference: Cesium.HeightReference.NONE,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 2.5, 1e8, 1.0),
            translucencyByDistance: new Cesium.NearFarScalar(5e4, 1.0, 1e8, 0.6),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        // Add a trail
        path: {
            resolution: 1,
            material: new Cesium.PolylineGlowMaterialProperty({
                glowPower: 0.4,
                color: Cesium.Color.YELLOW.withAlpha(0.9),
            }),
            width: 5,
            leadTime: 0.5,
            trailTime: 3.0,
        },
    });
    
    console.log(`✨ Created packet animation: ${packetId}, duration: ${duration}s, path segments: ${path.path.length}`);

    return {
        packetId,
        path,
        entity,
        startTime,
        endTime,
        duration,
    };
};

/**
 * Remove packet animation
 */
export const removePacketAnimation = (
    viewer: Cesium.Viewer,
    animation: PacketAnimation
): void => {
    viewer.entities.remove(animation.entity);
};


import type { UserTerminal, TerminalType, QoSRequirements, GenerateTerminalsRequest } from '../types/UserTerminalTypes';
import type { Position } from '../types/ModelTypes';
import type { NodeDTO } from '../types/NodeTypes';

interface Bounds {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
}

interface Region {
    name: string;
    bounds: Bounds;
}

/**
 * Generate a random number between min and max (inclusive)
 */
const random = (min: number, max: number): number => {
    return Math.random() * (max - min) + min;
};

/**
 * Generate a unique terminal ID
 */
const generateTerminalId = (index: number): string => {
    const timestamp = Date.now();
    return `TERM-${timestamp}-${index.toString().padStart(4, '0')}`;
};

/**
 * Generate random QoS requirements
 */
const generateRandomQoS = (): QoSRequirements => {
    const serviceTypes = ['VIDEO_STREAM', 'AUDIO_CALL', 'IMAGE_TRANSFER', 'TEXT_MESSAGE', 'FILE_TRANSFER'];
    const randomService = serviceTypes[Math.floor(Math.random() * serviceTypes.length)];

    return {
        maxLatencyMs: random(50, 500),
        minBandwidthMbps: random(1, 100),
        maxLossRate: random(0.001, 0.05),
        priority: Math.floor(random(1, 10)),
        serviceType: randomService,
    };
};

/**
 * Generate a random terminal type
 */
const generateRandomTerminalType = (): TerminalType => {
    const types: TerminalType[] = ['MOBILE', 'FIXED', 'VEHICLE', 'AIRCRAFT'];
    return types[Math.floor(Math.random() * types.length)];
};

/**
 * Generate a random position within bounds
 */
const generateRandomPosition = (bounds: Bounds): Position => {
    const latitude = random(bounds.minLat, bounds.maxLat);
    const longitude = random(bounds.minLon, bounds.maxLon);
    // Ground terminals: 0-100m altitude
    // Aircraft terminals: 5000-12000m altitude
    const altitude = Math.random() > 0.8 ? random(5000, 12000) : random(0, 100);

    return {
        latitude,
        longitude,
        altitude,
    };
};

/**
 * Check if a position is too close to existing nodes
 */
const isPositionValid = (
    position: Position,
    nodes: NodeDTO[],
    minDistanceKm: number = 1
): boolean => {
    const R = 6371; // Earth radius in km

    for (const node of nodes) {
        const nodePos = node.position;
        if (!nodePos) continue;

        const dLat = ((position.latitude - nodePos.latitude) * Math.PI) / 180;
        const dLon = ((position.longitude - nodePos.longitude) * Math.PI) / 180;

        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos((nodePos.latitude * Math.PI) / 180) *
                Math.cos((position.latitude * Math.PI) / 180) *
                Math.sin(dLon / 2) *
                Math.sin(dLon / 2);

        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        const distance = R * c;

        if (distance < minDistanceKm) {
            return false;
        }
    }

    return true;
};

/**
 * Generate random user terminals within specified bounds
 */
export const generateRandomUserTerminals = (
    count: number,
    bounds: Bounds,
    nodes: NodeDTO[] = [],
    maxAttempts: number = 100
): UserTerminal[] => {
    const terminals: UserTerminal[] = [];
    const terminalType = generateRandomTerminalType();

    for (let i = 0; i < count; i++) {
        let position: Position | null = null;
        let attempts = 0;

        // Try to find a valid position
        while (!position && attempts < maxAttempts) {
            const candidatePosition = generateRandomPosition(bounds);
            if (isPositionValid(candidatePosition, nodes)) {
                position = candidatePosition;
            }
            attempts++;
        }

        // If we couldn't find a valid position, use the last generated one anyway
        if (!position) {
            position = generateRandomPosition(bounds);
        }

        const terminalId = generateTerminalId(i);
        const terminal: UserTerminal = {
            id: terminalId,
            terminalId,
            terminalName: `Terminal ${i + 1}`,
            terminalType,
            position,
            status: 'idle',
            connectedNodeId: null,
            qosRequirements: generateRandomQoS(),
            metadata: {
                description: `Generated terminal ${i + 1}`,
                region: 'auto-generated',
            },
            lastUpdated: new Date().toISOString(),
        };

        terminals.push(terminal);
    }

    return terminals;
};

/**
 * Generate user terminals in a specific region with density
 */
export const generateUserTerminalsInRegion = (
    region: Region,
    density: number, // terminals per square degree (approximate)
    nodes: NodeDTO[] = []
): UserTerminal[] => {
    const area = (region.bounds.maxLat - region.bounds.minLat) * (region.bounds.maxLon - region.bounds.minLon);
    const count = Math.floor(area * density);
    return generateRandomUserTerminals(count, region.bounds, nodes);
};

/**
 * Validate terminal position against nodes
 */
export const validateTerminalPosition = (
    terminal: UserTerminal,
    nodes: NodeDTO[],
    minDistanceKm: number = 1
): boolean => {
    return isPositionValid(terminal.position, nodes, minDistanceKm);
};

/**
 * Generate terminals based on request parameters
 */
export const generateTerminalsFromRequest = (
    request: GenerateTerminalsRequest,
    nodes: NodeDTO[] = []
): UserTerminal[] => {
    if (request.region) {
        // If region is specified, use region-based generation
        // For now, we'll use default bounds if region is not a known region
        const defaultRegion: Region = {
            name: request.region,
            bounds: request.bounds || {
                minLat: -90,
                maxLat: 90,
                minLon: -180,
                maxLon: 180,
            },
        };
        return generateUserTerminalsInRegion(defaultRegion, request.density || 1, nodes);
    }

    // Use bounds-based generation
    const bounds = request.bounds || {
        minLat: -90,
        maxLat: 90,
        minLon: -180,
        maxLon: 180,
    };

    return generateRandomUserTerminals(request.count, bounds, nodes);
};


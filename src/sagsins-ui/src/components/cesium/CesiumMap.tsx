import { useEffect, useRef, useCallback } from "react";
import type { NodeInfo } from "../../types/node";

declare global {
  interface Window {
    Cesium?: any;
  }
}

const CESIUM_URL =
  "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.95.0/Cesium.js";
const CESIUM_CSS =
  "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.95.0/Widgets/widgets.css";

function loadCesiumScript(): Promise<typeof window.Cesium> {
  return new Promise((resolve, reject) => {
    if (window.Cesium) return resolve(window.Cesium);
    const script = document.createElement("script");
    script.src = CESIUM_URL;
    script.async = true;
    script.onload = () => resolve(window.Cesium);
    script.onerror = reject;
    document.body.appendChild(script);
  });
}

function loadCesiumCss() {
  if (document.getElementById("cesium-css")) return;
  const link = document.createElement("link");
  link.id = "cesium-css";
  link.rel = "stylesheet";
  link.href = CESIUM_CSS;
  document.head.appendChild(link);
}

// ====================== Utils ======================
function getNodeColor(nodeType: string): any {
  const Cesium = window.Cesium!;
  switch (nodeType) {
    case "SATELLITE":
      return Cesium.Color.GOLD;
    case "GROUND_STATION":
      return Cesium.Color.CYAN;
    case "UE":
      return Cesium.Color.LIME;
    case "RELAY":
      return Cesium.Color.ORANGE;
    case "SEA":
      return Cesium.Color.DODGERBLUE;
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

function createUEIcon(color: string): string {
  return `
    <svg width="20" height="32" xmlns="http://www.w3.org/2000/svg">
      <rect x="4" y="2" width="12" height="26" rx="3" ry="3"
        fill="${color}" stroke="black" stroke-width="1"/>
      <circle cx="10" cy="26" r="2" fill="black"/>
    </svg>`;
}

function createRelayIcon(color: string): string {
  return `
    <svg width="28" height="28" xmlns="http://www.w3.org/2000/svg">
      <circle cx="14" cy="14" r="10" fill="${color}" opacity="0.6"/>
      <path d="M10 10 L18 14 L10 18 Z" fill="white"/>
    </svg>`;
}

function createSeaIcon(color: string): string {
  return `
    <svg width="32" height="24" xmlns="http://www.w3.org/2000/svg">
      <rect x="4" y="10" width="24" height="8" fill="${color}" stroke="black" stroke-width="1"/>
      <polygon points="4,10 16,2 28,10" fill="${color}" stroke="black" stroke-width="1"/>
    </svg>`;
}

interface CesiumMapProps {
  nodes: NodeInfo[];
  selectedNodeId?: string;
  onNodeFocus?: (nodeId: string) => void;
}

function CesiumMap({ nodes, selectedNodeId, onNodeFocus }: CesiumMapProps) {
  const cesiumRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    loadCesiumCss();
    loadCesiumScript().then((Cesium) => {
      window.Cesium = Cesium;

      Cesium.Ion.defaultAccessToken =
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJjYTNkNmJmMC03MWJlLTRhZDgtYjI1My1jMDBjNDg0Y2NjMGUiLCJpZCI6MzM4ODk5LCJpYXQiOjE3NTg2MzMwNzN9.QLsbfWFGA6TesQEJBl4ktgCRve3LdQMk2BXTZB7fkWM";

      const viewer = new Cesium.Viewer(cesiumRef.current!, {
        terrainProvider: Cesium.createWorldTerrain(),
        homeButton: true,
        sceneModePicker: true,
        baseLayerPicker: true,
        navigationHelpButton: true,
        animation: true,
        timeline: true,
        fullscreenButton: true,
      });

      viewerRef.current = viewer;

      // Camera ban đầu
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000),
        orientation: {
          heading: 0,
          pitch: -Cesium.Math.PI_OVER_TWO,
          roll: 0,
        },
      });

      // Click handler
      viewer.cesiumWidget.screenSpaceEventHandler.setInputAction(
        (event: any) => {
          const pickedObject = viewer.scene.pick(event.position);
          if (pickedObject && pickedObject.id) {
            onNodeFocus?.(pickedObject.id.id);
          }
        },
        Cesium.ScreenSpaceEventType.LEFT_CLICK
      );

      return () => {
        if (viewer && !viewer.isDestroyed()) {
          viewer.destroy();
        }
      };
    });
  }, [onNodeFocus]);

  const addNode = useCallback(
    (node: NodeInfo) => {
      const Cesium = window.Cesium;
      if (!viewerRef.current || !Cesium) return;

      const viewer = viewerRef.current;
      const { nodeId, nodeType, position } = node;
      const lon = Number(position?.longitude);
      const lat = Number(position?.latitude);
      const alt = Number(position?.altitude) || 1000;
      const color = getNodeColor(nodeType);

      if (
        isNaN(lon) ||
        isNaN(lat) ||
        lon < -180 ||
        lon > 180 ||
        lat < -90 ||
        lat > 90
      ) {
        console.warn("❌ Node dữ liệu không hợp lệ:", node);
        return;
      }

      const entityOptions: any = {
        id: nodeId,
        name: `${nodeType}: ${nodeId}`,
        position: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
      };

      const isSelected = selectedNodeId === nodeId;

      let iconSvg = "";
      switch (nodeType) {
        case "SATELLITE":
          iconSvg = createSatelliteIcon(color.toCssColorString());
          break;
        case "GROUND_STATION":
          iconSvg = createGroundStationIcon(color.toCssColorString());
          break;
        case "UE":
          iconSvg = createUEIcon(color.toCssColorString());
          break;
        case "RELAY":
          iconSvg = createRelayIcon(color.toCssColorString());
          break;
        case "SEA":
          iconSvg = createSeaIcon(color.toCssColorString());
          break;
        default:
          iconSvg = createSatelliteIcon("#ffffff");
      }

      entityOptions.billboard = {
        image: "data:image/svg+xml;base64," + btoa(iconSvg),
        scale: isSelected ? 1.1 : 0.9,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      };

      entityOptions.label = {
        text: nodeId,
        font: "10px monospace",
        fillColor: color,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        verticalOrigin: Cesium.VerticalOrigin.TOP,
        pixelOffset: new Cesium.Cartesian2(0, -18),
        distanceDisplayCondition:
          nodeType === "SATELLITE"
            ? undefined
            : new Cesium.DistanceDisplayCondition(0, 5_000_000),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      };

      return viewer.entities.add(entityOptions);
    },
    [selectedNodeId]
  );

  useEffect(() => {
    if (!viewerRef.current) return;
    const viewer = viewerRef.current;
    viewer.entities.removeAll();
    nodes.forEach((node) => addNode(node));
  }, [nodes, selectedNodeId, addNode]);

  useEffect(() => {
    if (!viewerRef.current || !selectedNodeId) return;
    const viewer = viewerRef.current;
    const entity = viewer.entities.getById(selectedNodeId);
    if (entity) {
      viewer.flyTo(entity, {
        duration: 2.0,
        offset: new window.Cesium.HeadingPitchRange(0, -0.5, 1000000),
      });
    }
  }, [selectedNodeId]);

  return <div ref={cesiumRef} className="w-full h-full absolute inset-0" />;
}

export { CesiumMap };
export default CesiumMap;

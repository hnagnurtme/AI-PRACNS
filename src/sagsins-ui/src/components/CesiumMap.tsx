import { useEffect, useRef } from "react";
import type { NodeInfo } from "../types/node";

declare global {
  interface Window {
    Cesium?: any;
  }
}

const CESIUM_URL = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.95.0/Cesium.js";
const CESIUM_CSS = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.95.0/Widgets/widgets.css";

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

export default function CesiumMap({ nodes }: { nodes: NodeInfo[] }) {
  const cesiumRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadCesiumCss();
    loadCesiumScript().then((Cesium) => {
      Cesium.Ion.defaultAccessToken =
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJjYTNkNmJmMC03MWJlLTRhZDgtYjI1My1jMDBjNDg0Y2NjMGUiLCJpZCI6MzM4ODk5LCJpYXQiOjE3NTg2MzMwNzN9.QLsbfWFGA6TesQEJBl4ktgCRve3LdQMk2BXTZB7fkWM";
      const viewer = new Cesium.Viewer(cesiumRef.current, {
        terrainProvider: Cesium.createWorldTerrain(),
        homeButton: true,
        sceneModePicker: true,
        baseLayerPicker: true,
        navigationHelpButton: true,
        animation: true,
        timeline: true,
        fullscreenButton: true,
      });
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000),
        orientation: {
          heading: 0,
          pitch: -Cesium.Math.PI_OVER_TWO,
          roll: 0,
        },
      });
      // Clean up previous entities
      viewer.entities.removeAll();
      nodes.forEach((node: NodeInfo) => {
        const lon = Number(node?.position?.longitude);
        const lat = Number(node?.position?.latitude);
        const alt = Number(node?.position?.altitude) || 1000;
        if (isNaN(lon) || isNaN(lat)) return;
        viewer.entities.add({
          id: node.nodeId,
          name: `${node.nodeType}: ${node.nodeId}`,
          position: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
          point: {
            pixelSize: 8,
            color: Cesium.Color.GOLD,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 2,
          },
          label: {
            text: node.nodeId,
            font: "10px monospace",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            verticalOrigin: Cesium.VerticalOrigin.TOP,
            pixelOffset: new Cesium.Cartesian2(0, -20),
          },
        });
      });
    });
  }, [nodes]);

  return <div ref={cesiumRef} className="w-full h-full" />;
}

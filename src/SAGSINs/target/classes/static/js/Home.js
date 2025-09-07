// ====================== Cesium Config ======================
Cesium.Ion.defaultAccessToken = "";


const viewer = new Cesium.Viewer("cesiumContainer", {
  terrainProvider: Cesium.createWorldTerrain(),
  homeButton: true,
  sceneModePicker: true,
  baseLayerPicker: true,
  navigationHelpButton: true,
  animation: true,
  timeline: true,
  fullscreenButton: true,
});

// Camera ban đầu
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000),
  orientation: {
    heading: 0,
    pitch: -Cesium.Math.PI_OVER_TWO,
    roll: 0,
  },
});

// ====================== Utils ======================
function getNodeColor(nodeType) {
  switch (nodeType) {
    case "SATELLITE": return Cesium.Color.GOLD;
    case "GROUND_STATION": return Cesium.Color.CYAN;
    case "UE": return Cesium.Color.LIME;
    case "RELAY": return Cesium.Color.ORANGE;
    case "SEA": return Cesium.Color.DODGERBLUE;
    default: return Cesium.Color.WHITE;
  }
}

function createSatelliteIcon(color) {
  return `
    <svg width="32" height="32" xmlns="http://www.w3.org/2000/svg">
      <circle cx="16" cy="16" r="6" fill="${color}" stroke="#000" stroke-width="1"/>
      <rect x="13" y="8" width="6" height="2" fill="${color}" opacity="0.8"/>
      <rect x="13" y="22" width="6" height="2" fill="${color}" opacity="0.8"/>
      <rect x="8" y="13" width="2" height="6" fill="${color}" opacity="0.8"/>
      <rect x="22" y="13" width="2" height="6" fill="${color}" opacity="0.8"/>
    </svg>`;
}

function addNode(node) {
  const { nodeId, nodeType, position, orbit, velocity } = node;
  const lon = Number(position?.longitude);
  const lat = Number(position?.latitude);
  const alt = Number(position?.altitude) || 1000;
  const color = getNodeColor(nodeType);

  if (isNaN(lon) || isNaN(lat) || lon < -180 || lon > 180 || lat < -90 || lat > 90) {
    console.warn("❌ Node dữ liệu không hợp lệ:", node);
    return;
  }

  // Nếu là vệ tinh thì animate theo orbit
  let positionProperty;
  if (nodeType === "SATELLITE" && orbit) {
    const startTime = Cesium.JulianDate.now();
    const endTime = Cesium.JulianDate.addDays(startTime, 1, new Cesium.JulianDate());

    positionProperty = new Cesium.SampledPositionProperty();
    positionProperty.setInterpolationOptions({
      interpolationDegree: 5,
      interpolationAlgorithm: Cesium.LagrangePolynomialApproximation,
    });

    const totalSeconds = 86400;
    const sampleInterval = 60;
    for (let i = 0; i <= totalSeconds; i += sampleInterval) {
      const time = Cesium.JulianDate.addSeconds(startTime, i, new Cesium.JulianDate());
      const angle = i * (velocity?.speed || 0.001);
      const longitude = (Cesium.Math.toDegrees(angle) % 360) - 180;
      const latitude = Math.sin(angle) * (orbit?.inclination || 0);
      const pos = Cesium.Cartesian3.fromDegrees(longitude, latitude, orbit?.altitude || alt);
      positionProperty.addSample(time, pos);
    }
  }

  const entityOptions = {
    id: nodeId,
    name: `${nodeType}: ${nodeId}`,
    position: positionProperty || Cesium.Cartesian3.fromDegrees(lon, lat, alt),
  };

  // CHỈ SATELLITE có icon, các node khác là chấm nhỏ
  if (nodeType === "SATELLITE") {
    entityOptions.billboard = {
      image: "data:image/svg+xml;base64," + btoa(createSatelliteIcon(color.toCssColorString())),
      scale: 0.8,
      pixelOffset: new Cesium.Cartesian2(0, 0),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    };
    
    // Label cho satellite - luôn hiện
    entityOptions.label = {
      text: nodeId,
      font: "10px monospace",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      verticalOrigin: Cesium.VerticalOrigin.TOP,
      pixelOffset: new Cesium.Cartesian2(0, -20),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    };
  } else {
    // Các node khác chỉ là chấm nhỏ
    entityOptions.point = {
      pixelSize: 6,
      color: color,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 1,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    };
    
    // Label chỉ hiện khi zoom gần (distance-based)
    entityOptions.label = {
      text: nodeId,
      font: "9px monospace",
      fillColor: color,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 1,
      verticalOrigin: Cesium.VerticalOrigin.TOP,
      pixelOffset: new Cesium.Cartesian2(0, -12),
      // Chỉ hiện label khi camera gần (< 5 triệu mét)
      distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5000000),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    };
  }

  return viewer.entities.add(entityOptions);
}

// ====================== Khởi tạo ======================
setTimeout(() => {
  try {
    document.getElementById("loadingScreen").style.display = "none";
    document.getElementById("controlPanel").style.opacity = "1";
    document.getElementById("controlPanel").style.animation = "slideInLeft 0.8s ease-out";

    // Clock config
    const startTime = Cesium.JulianDate.now();
    const endTime = Cesium.JulianDate.addDays(startTime, 1, new Cesium.JulianDate());
    viewer.clock.startTime = startTime;
    viewer.clock.stopTime = endTime;
    viewer.clock.currentTime = startTime;
    viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
    viewer.clock.shouldAnimate = true;
    viewer.clock.multiplier = 50;

    // 🛰️ Load từ backend (ví dụ mock)
    if(Array.isArray(nodes) === false) {
      throw new Error("Dữ liệu nodes không hợp lệ");
    }
    if(nodes.length === 0) {
      throw new Error("Không có node nào để hiển thị");
    }
    
    console.log("🚀 Đang tải", nodes.length, "node...");
    
    // Đếm số lượng từng loại node
    const nodeCount = {};
    nodes.forEach(node => {
      const type = node.nodeType || 'UNKNOWN';
      nodeCount[type] = (nodeCount[type] || 0) + 1;
      addNode(node);
    });
    
    console.log("📊 Thống kê nodes:", nodeCount);
    console.log("✅ Ứng dụng đã sẵn sàng!");
    
    // Thông báo hướng dẫn
    viewer.cesiumWidget.showInfoBox = true;
    setTimeout(() => {
      console.log("💡 Hướng dẫn: Zoom gần để xem label của Ground Station, UE, Relay và Sea nodes");
    }, 2000);
    
  } catch (error) {
    console.error("❌ Lỗi:", error);
    document.getElementById("loadingScreen").innerHTML = `
      <div style="color: #ef4444;">
        <h3>❌ Có lỗi xảy ra</h3>
        <p>${error.message}</p>
      </div>
    `;
  }
}, 1000);
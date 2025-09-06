// Home.js - CesiumJS logic for Satellite WebView

// Cấu hình Cesium
Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1NTUwMWM1NS1iY2YxLTRjOTUtODk3My02ZTIzMDc0MTNhMjQiLCJpZCI6MzM4ODk5LCJpYXQiOjE3NTcxNDI4MDV9.YgKJ7HbZeS00sK8d01sclHgnwkpGOP64Fl0V1Oq11DI";

// Khởi tạo viewer với cấu hình tối ưu
const viewer = new Cesium.Viewer('cesiumContainer', {
  terrainProvider: Cesium.createWorldTerrain(),
  homeButton: true,
  sceneModePicker: true,
  baseLayerPicker: true,
  navigationHelpButton: true,
  animation: true,
  timeline: true,
  fullscreenButton: true,
  vrButton: true
});

// Cài đặt camera ban đầu
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(0, 0, 15000000),
  orientation: {
    heading: 0,
    pitch: -Cesium.Math.PI_OVER_TWO,
    roll: 0
  }
});

// Thêm code vệ tinh với SampledPositionProperty để tránh lỗi getValueInReferenceFrame
function addOrbitingSatellite(name, color, altitude, speed, inclination, startAngle = 0) {
  const startTime = Cesium.JulianDate.now();
  const endTime = Cesium.JulianDate.addDays(startTime, 1, new Cesium.JulianDate());

  // Sử dụng SampledPositionProperty thay vì CallbackProperty
  const positionProperty = new Cesium.SampledPositionProperty();
  positionProperty.setInterpolationOptions({
    interpolationDegree: 5,
    interpolationAlgorithm: Cesium.LagrangePolynomialApproximation
  });

  // Tạo các điểm mẫu cho quỹ đạo
  const totalSeconds = 86400; // 24 giờ
  const sampleInterval = 60; // mỗi 60 giây một điểm
  for (let i = 0; i <= totalSeconds; i += sampleInterval) {
    const time = Cesium.JulianDate.addSeconds(startTime, i, new Cesium.JulianDate());
    const angle = startAngle + (i * speed);
    const longitude = Cesium.Math.toDegrees(angle) % 360;
    const latitude = Math.sin(angle) * inclination;
    const position = Cesium.Cartesian3.fromDegrees(longitude, latitude, altitude);
    positionProperty.addSample(time, position);
  }

  // Cấu hình availability interval
  const availability = new Cesium.TimeIntervalCollection([
    new Cesium.TimeInterval({
      start: startTime,
      stop: endTime
    })
  ]);

  return viewer.entities.add({
    name: name,
    availability: availability,
    position: positionProperty,
    billboard: {
      image: 'data:image/svg+xml;base64,' + btoa(`
        <svg width="40" height="40" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="grad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" style="stop-color:${color.toCssColorString()};stop-opacity:1" />
              <stop offset="100%" style="stop-color:${color.toCssColorString()};stop-opacity:0.3" />
            </radialGradient>
          </defs>
          <circle cx="20" cy="20" r="8" fill="url(#grad)" stroke="${color.toCssColorString()}" stroke-width="2"/>
          <rect x="16" y="10" width="8" height="3" fill="${color.toCssColorString()}" opacity="0.7"/>
          <rect x="16" y="27" width="8" height="3" fill="${color.toCssColorString()}" opacity="0.7"/>
          <rect x="10" y="16" width="3" height="8" fill="${color.toCssColorString()}" opacity="0.7"/>
          <rect x="27" y="16" width="3" height="8" fill="${color.toCssColorString()}" opacity="0.7"/>
        </svg>`),
      scale: 1.0,
      pixelOffset: new Cesium.Cartesian2(0, 0),
      heightReference: Cesium.HeightReference.NONE
    },
    label: {
      text: name,
      font: "14px bold Arial",
      fillColor: color,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      verticalOrigin: Cesium.VerticalOrigin.TOP,
      pixelOffset: new Cesium.Cartesian2(0, -30),
      disableDepthTestDistance: Number.POSITIVE_INFINITY
    }
  });
}

// Khởi tạo vệ tinh sau 1 giây
setTimeout(() => {
  try {
    // Xóa loading screen
    document.getElementById('loadingScreen').style.display = 'none';
    document.getElementById('controlPanel').style.opacity = '1';
    document.getElementById('controlPanel').style.animation = 'slideInLeft 0.8s ease-out';

    // Tạo vệ tinh với thời gian đã định
    const startTime = Cesium.JulianDate.now();
    const endTime = Cesium.JulianDate.addDays(startTime, 1, new Cesium.JulianDate());
    viewer.clock.startTime = startTime;
    viewer.clock.stopTime = endTime;
    viewer.clock.currentTime = startTime;
    viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;

    const satA = addOrbitingSatellite(
      "🛰️ ISS-Demo",
      Cesium.Color.GOLD,
      500000, 0.003, 15, 0
    );
    const satB = addOrbitingSatellite(
      "📡 GPS-Demo",
      Cesium.Color.CYAN,
      1200000, 0.002, 45, 2.09
    );
    const satC = addOrbitingSatellite(
      "🌍 GEO-Demo",
      Cesium.Color.LIME,
      2000000, 0.001, 60, 4.19
    );
    viewer.clock.shouldAnimate = true;
    viewer.clock.multiplier = 50;
    console.log("✅ Ứng dụng đã sẵn sàng!");
  } catch (error) {
    console.error("❌ Lỗi:", error);
    document.getElementById('loadingScreen').innerHTML = `
      <div style="color: #ef4444;">
        <h3>❌ Có lỗi xảy ra</h3>
        <p>${error.message}</p>
      </div>
    `;
  }
}, 1000);

// Thêm sự kiện click vào satellite cards
if (document.querySelectorAll('.satellite-card').length > 0) {
  document.querySelectorAll('.satellite-card').forEach((card, index) => {
    card.addEventListener('click', () => {
      const satellites = viewer.entities.values;
      if (satellites[index]) {
        viewer.trackedEntity = satellites[index];
        viewer.camera.zoomTo(satellites[index], new Cesium.HeadingPitchRange(0, -Math.PI / 4, 1000000));
      }
    });
  });
}

// Click chọn vệ tinh functionality
let entity;
viewer.screenSpaceEventHandler.setInputAction(function(click) {
  const cartesian = viewer.scene.pickPosition(click.position);
  if (Cesium.defined(cartesian)) {
    const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
    const lat = Cesium.Math.toDegrees(cartographic.latitude).toFixed(6);
    const lon = Cesium.Math.toDegrees(cartographic.longitude).toFixed(6);
    const height = cartographic.height.toFixed(2);
    if (entity) viewer.entities.remove(entity);
    entity = viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(lon, lat, height),
      point: { pixelSize: 12, color: Cesium.Color.RED }
    });
    fetch(`/setPosition?lat=${lat}&lng=${lon}&alt=${height}`)
      .then(res => res.text())
      .then(msg => console.log(msg))
      .catch(err => console.error('Position update error:', err));
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

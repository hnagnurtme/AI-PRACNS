package com.sagin.satellite.view;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import org.slf4j.Logger;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

public class SatelliteWebView {
    private static final Logger logger = org.slf4j.LoggerFactory.getLogger(SatelliteWebView.class);

    public static void startServer(int port) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);

        // Serve HTML page
        server.createContext("/", new HttpHandler() {
            @Override
            public void handle(HttpExchange exchange) throws IOException {
                String html = """
                <!DOCTYPE html>
                <html lang="vi">
                <head>
                  <meta charset="utf-8">
                  <meta name="viewport" content="width=device-width, initial-scale=1">
                
                  <!-- Cesium CSS -->
                  <link href="https://cdnjs.cloudflare.com/ajax/libs/cesium/1.95.0/Widgets/widgets.css" rel="stylesheet">
                  
                  <style>
                    * {
                      margin: 0;
                      padding: 0;
                      box-sizing: border-box;
                    }

                    body {
                      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                      background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
                      overflow: hidden;
                      position: relative;
                    }

                    /* Container chính */
                    #cesiumContainer {
                      width: 100vw;
                      height: 100vh;
                      position: relative;
                      box-shadow: inset 0 0 100px rgba(0, 255, 255, 0.1);
                    }

                    /* Header */
                    .header {
                      position: absolute;
                      top: 20px;
                      left: 20px;
                      right: 20px;
                      z-index: 1000;
                      background: rgba(15, 23, 42, 0.95);
                      backdrop-filter: blur(20px);
                      border: 1px solid rgba(59, 130, 246, 0.3);
                      border-radius: 16px;
                      padding: 20px 25px;
                      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    }

                    .header:hover {
                      transform: translateY(-2px);
                      box-shadow: 0 12px 48px rgba(59, 130, 246, 0.15);
                      border-color: rgba(59, 130, 246, 0.5);
                    }

                    .header h1 {
                      color: #ffffff;
                      font-size: 24px;
                      font-weight: 700;
                      margin-bottom: 8px;
                      display: flex;
                      align-items: center;
                      gap: 12px;
                    }

                    .header p {
                      color: #94a3b8;
                      font-size: 14px;
                      line-height: 1.5;
                    }

                    /* Panel điều khiển */
                    .control-panel {
                      position: absolute;
                      top: 120px;
                      left: 20px;
                      z-index: 1000;
                      background: rgba(15, 23, 42, 0.95);
                      backdrop-filter: blur(20px);
                      border: 1px solid rgba(59, 130, 246, 0.3);
                      border-radius: 16px;
                      padding: 20px;
                      min-width: 280px;
                      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                      animation: slideInLeft 0.8s ease-out;
                    }

                    @keyframes slideInLeft {
                      from {
                        opacity: 0;
                        transform: translateX(-100px);
                      }
                      to {
                        opacity: 1;
                        transform: translateX(0);
                      }
                    }

                    .control-panel h3 {
                      color: #ffffff;
                      font-size: 18px;
                      font-weight: 600;
                      margin-bottom: 16px;
                      display: flex;
                      align-items: center;
                      gap: 8px;
                    }

                    /* Satellite cards */
                    .satellite-card {
                      background: rgba(30, 41, 59, 0.8);
                      border: 1px solid rgba(71, 85, 105, 0.4);
                      border-radius: 12px;
                      padding: 16px;
                      margin-bottom: 12px;
                      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                      cursor: pointer;
                    }

                    .satellite-card:hover {
                      transform: translateY(-2px) scale(1.02);
                      border-color: rgba(59, 130, 246, 0.6);
                      box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
                    }

                    .satellite-info {
                      display: flex;
                      align-items: center;
                      gap: 12px;
                      margin-bottom: 8px;
                    }

                    .satellite-icon {
                      width: 32px;
                      height: 32px;
                      border-radius: 50%;
                      display: flex;
                      align-items: center;
                      justify-content: center;
                      font-size: 16px;
                      animation: rotate 10s linear infinite;
                    }

                    @keyframes rotate {
                      from { transform: rotate(0deg); }
                      to { transform: rotate(360deg); }
                    }

                    .satellite-details h4 {
                      color: #ffffff;
                      font-size: 14px;
                      font-weight: 600;
                      margin-bottom: 2px;
                    }

                    .satellite-details p {
                      color: #64748b;
                      font-size: 12px;
                    }

                    .satellite-stats {
                      display: grid;
                      grid-template-columns: 1fr 1fr;
                      gap: 8px;
                      margin-top: 8px;
                    }

                    .stat-item {
                      background: rgba(51, 65, 85, 0.5);
                      border-radius: 8px;
                      padding: 8px;
                      text-align: center;
                    }

                    .stat-label {
                      color: #64748b;
                      font-size: 10px;
                      margin-bottom: 2px;
                    }

                    .stat-value {
                      color: #ffffff;
                      font-size: 12px;
                      font-weight: 600;
                    }

                    /* Loading animation */
                    .loading {
                      position: absolute;
                      top: 50%;
                      left: 50%;
                      transform: translate(-50%, -50%);
                      z-index: 2000;
                      color: #ffffff;
                      text-align: center;
                      animation: fadeIn 0.5s ease-out;
                    }

                    @keyframes fadeIn {
                      from { opacity: 0; }
                      to { opacity: 1; }
                    }

                    .spinner {
                      width: 40px;
                      height: 40px;
                      border: 3px solid rgba(59, 130, 246, 0.3);
                      border-top-color: #3b82f6;
                      border-radius: 50%;
                      animation: spin 1s linear infinite;
                      margin: 0 auto 16px;
                    }

                    @keyframes spin {
                      to { transform: rotate(360deg); }
                    }

                    /* Responsive */
                    @media (max-width: 768px) {
                      .header {
                        left: 10px;
                        right: 10px;
                        padding: 16px 20px;
                      }
                      
                      .header h1 {
                        font-size: 20px;
                      }
                      
                      .control-panel {
                        left: 10px;
                        min-width: 260px;
                      }
                    }

                    /* Custom Cesium widget styling */
                    .cesium-widget-credits {
                      display: none !important;
                    }

                    .cesium-timeline-main {
                      background: rgba(15, 23, 42, 0.95) !important;
                      border-top: 1px solid rgba(59, 130, 246, 0.3) !important;
                    }

                    .cesium-animation-widget {
                      background: rgba(15, 23, 42, 0.95) !important;
                      border: 1px solid rgba(59, 130, 246, 0.3) !important;
                      border-radius: 8px !important;
                    }

                    .cesium-viewer-toolbar {
                      background: rgba(15, 23, 42, 0.95) !important;
                      border-radius: 8px !important;
                      border: 1px solid rgba(59, 130, 246, 0.3) !important;
                    }

                    .cesium-button {
                      background: rgba(30, 41, 59, 0.8) !important;
                      color: #ffffff !important;
                      border: 1px solid rgba(71, 85, 105, 0.4) !important;
                    }

                    .cesium-button:hover {
                      background: rgba(59, 130, 246, 0.2) !important;
                      border-color: rgba(59, 130, 246, 0.6) !important;
                    }
                  </style>
                </head>

                <body>
                  <!-- Loading screen -->
                  <div class="loading" id="loadingScreen">
                    <div class="spinner"></div>
                    <h3>🚀 Đang khởi tạo vệ tinh...</h3>
                    <p>Vui lòng chờ trong giây lát</p>
                  </div>


                  <!-- Control Panel -->
                  <div class="control-panel" id="controlPanel" style="opacity: 0;">
                    <h3>
                      <span>📡</span>
                      Danh sách vệ tinh
                    </h3>

                    <div class="satellite-card" style="border-left: 4px solid #ffd700;">
                      <div class="satellite-info">
                        <div class="satellite-icon" style="background: rgba(255, 215, 0, 0.2); color: #ffd700;">
                          🛰️
                        </div>
                        <div class="satellite-details">
                          <h4>ISS-Demo</h4>
                          <p>Trạm vũ trụ quốc tế</p>
                        </div>
                      </div>
                      <div class="satellite-stats">
                        <div class="stat-item">
                          <div class="stat-label">Độ cao</div>
                          <div class="stat-value">500 km</div>
                        </div>
                        <div class="stat-item">
                          <div class="stat-label">Tốc độ</div>
                          <div class="stat-value">Nhanh</div>
                        </div>
                      </div>
                    </div>

                    <div class="satellite-card" style="border-left: 4px solid #00ffff;">
                      <div class="satellite-info">
                        <div class="satellite-icon" style="background: rgba(0, 255, 255, 0.2); color: #00ffff;">
                          📡
                        </div>
                        <div class="satellite-details">
                          <h4>GPS-Demo</h4>
                          <p>Hệ thống định vị toàn cầu</p>
                        </div>
                      </div>
                      <div class="satellite-stats">
                        <div class="stat-item">
                          <div class="stat-label">Độ cao</div>
                          <div class="stat-value">1200 km</div>
                        </div>
                        <div class="stat-item">
                          <div class="stat-label">Tốc độ</div>
                          <div class="stat-value">Vừa</div>
                        </div>
                      </div>
                    </div>

                    <div class="satellite-card" style="border-left: 4px solid #00ff00;">
                      <div class="satellite-info">
                        <div class="satellite-icon" style="background: rgba(0, 255, 0, 0.2); color: #00ff00;">
                          🌍
                        </div>
                        <div class="satellite-details">
                          <h4>GEO-Demo</h4>
                          <p>Vệ tinh địa tĩnh</p>
                        </div>
                      </div>
                      <div class="satellite-stats">
                        <div class="stat-item">
                          <div class="stat-label">Độ cao</div>
                          <div class="stat-value">2000 km</div>
                        </div>
                        <div class="stat-item">
                          <div class="stat-label">Tốc độ</div>
                          <div class="stat-value">Chậm</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <!-- Cesium Container -->
                  <div id="cesiumContainer"></div>

                  <!-- Cesium JS -->
                  <script src="https://cdnjs.cloudflare.com/ajax/libs/cesium/1.95.0/Cesium.js"></script>
                  
                  <script>
                    // Cấu hình Cesium
                    Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1NTUwMWM1NS1iY2YxLTRjOTUtODk3My02ZTIzMDc0MTNhMjQiLCJpZCI6MzM4ODk5LCJpYXQiOjE3NTcxNDI4MDV9.YgKJ7HbZeS00sK8d01sclHgnwkpGOP64Fl0V1Oq11DI";

                    // Khởi tạo viewer với cấu hình tối ưu
                    const viewer = new Cesium.Viewer('cesiumContainer', {
                      terrainProvider: Cesium.createWorldTerrain(),
                      homeButton: true,
                      sceneModePicker: true,
                      baseLayerPicker: false,
                      navigationHelpButton: false,
                      animation: true,
                      timeline: true,
                      fullscreenButton: true,
                      vrButton: false
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
        
        // Cấu hình clock với thời gian xác định
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
        viewer.clock.multiplier = 50;                        console.log("✅ Ứng dụng đã sẵn sàng!");

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
                    document.querySelectorAll('.satellite-card').forEach((card, index) => {
                      card.addEventListener('click', () => {
                        const satellites = viewer.entities.values;
                        if (satellites[index]) {
                          viewer.trackedEntity = satellites[index];
                          viewer.camera.zoomTo(satellites[index], new Cesium.HeadingPitchRange(0, -Math.PI / 4, 1000000));
                        }
                      });
                    });
                    
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
                  </script>
                </body>
                </html>
                """;

                exchange.getResponseHeaders().add("Content-Type", "text/html; charset=UTF-8");
                exchange.sendResponseHeaders(200, html.getBytes().length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(html.getBytes());
                }
            }
        });

        // API nhận tọa độ từ frontend
        server.createContext("/setPosition", new HttpHandler() {
            @Override
            public void handle(HttpExchange exchange) throws IOException {
                String query = exchange.getRequestURI().getQuery();
                System.out.println("Received position: " + query);

                String response = "Position set: " + query;
                exchange.sendResponseHeaders(200, response.getBytes().length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(response.getBytes());
                }
            }
        });

        server.setExecutor(null);
        server.start();
        logger.info("Satellite WebView (CesiumJS) started on port {}", port);
    }
}

// src/map/CesiumConfig.ts

import * as Cesium from 'cesium';

interface BuildModuleUrl {
    ( relativeUrl: string ): string; // Kiểu gốc của hàm
    setBaseUrl ( url: string ): void; // Phương thức chúng ta cần
}
// Khai báo Interface tạm thời cho các thuộc tính bị thiếu
interface CesiumCreditDisplay extends Cesium.CreditDisplay {
    cesiumWidget: boolean;
}

interface CesiumScene extends Cesium.Scene {
    useMSAA: boolean;
}

/**
 * Khởi tạo cấu hình toàn cục cho CesiumJS.
 * Cần được gọi một lần duy nhất trong main.tsx hoặc App.tsx.
 * @param cesiumBaseUrl Đường dẫn đến thư mục Cesium Assets trong thư mục Public (Ví dụ: '/Cesium/')
 */
export const setupCesiumConfig = ( cesiumBaseUrl: string ): void => {
    // 1. Cấu hình đường dẫn tài nguyên (Assets, Workers, v.v.)
    // Đây là bước quan trọng nhất để Cesium tải các file cần thiết.
    const buildModuleUrl = Cesium.buildModuleUrl as unknown as BuildModuleUrl;

    buildModuleUrl.setBaseUrl( cesiumBaseUrl );
    // 2. Cấu hình Token (Nếu bạn dùng Cesium ion cho các lớp nền đặc biệt)
    // Nếu bạn chỉ dùng lớp nền mặc định (Bing Maps/OpenStreetMap), bạn có thể bỏ qua.
    if ( import.meta.env.VITE_CESIUM_ION_TOKEN ) {
        Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN;
    }
    Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJjYTNkNmJmMC03MWJlLTRhZDgtYjI1My1jMDBjNDg0Y2NjMGUiLCJpZCI6MzM4ODk5LCJpYXQiOjE3NTg2MzMwNzN9.QLsbfWFGA6TesQEJBl4ktgCRve3LdQMk2BXTZB7fkWM";

    // 3. Tắt hộp thông tin mặc định (Nếu bạn muốn dùng UI React tùy chỉnh)
    // Việc này giúp tránh xung đột với NodeDetailCard của bạn.
    ( Cesium.CreditDisplay as unknown as CesiumCreditDisplay ).cesiumWidget = false;

    // 4. Thiết lập chế độ làm mịn (Anti-aliasing) cho chất lượng đồ họa tốt hơn
    ( Cesium.Scene.prototype as unknown as CesiumScene ).useMSAA = true;

    console.info( `[Cesium Config] Base URL set to: ${ cesiumBaseUrl }` );
};

/**
 * Định nghĩa các hằng số màu sắc cho các Node Entities.
 */
export const NODE_COLORS = {
    HEALTHY: Cesium.Color.GREEN,
    UNHEALTHY: Cesium.Color.RED,
    OUTLINE: Cesium.Color.BLACK,
};

/**
 * Cấu hình Camera View mặc định (Ví dụ: nhìn xuống từ quỹ đạo hoặc tập trung vào một khu vực).
 */
export const DEFAULT_CAMERA_VIEW = {
    longitude: 105.0,  // Việt Nam
    latitude: 15.0,
    height: 8000000.0,
};

/**
 * Hàm tiện ích để bay (fly) camera đến một vị trí Node.
 * @param viewer Instance của Cesium Viewer
 * @param longitude Kinh độ
 * @param latitude Vĩ độ
 * @param altitude Độ cao (mét)
 */
export const flyToLocation = (
    viewer: Cesium.Viewer,
    longitude: number,
    latitude: number,
    altitude: number = 2000000
): void => {

    viewer.camera.flyTo( {
        destination: Cesium.Cartesian3.fromDegrees( longitude, latitude, altitude ),
        duration: 3,
        maximumHeight: 30000000, // bay cao tối đa 30.000km
        orientation: {
            heading: 0,
            pitch: Cesium.Math.toRadians( -90.0 ),
            roll: 0
        }
    } );
};
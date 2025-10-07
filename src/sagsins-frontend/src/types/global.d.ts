
import * as Cesium from 'cesium';

declare global {
    /**
     * Mở rộng đối tượng Window để thêm thuộc tính 'viewer' (tùy chọn).
     */
    interface Window {
        viewer?: Cesium.Viewer; 
    }
}

// KHÔNG CẦN THÊM BẤT KỲ MÃ NÀO KHÁC TRONG FILE NÀY
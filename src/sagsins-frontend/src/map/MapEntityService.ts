import * as Cesium from 'cesium';
import type { NodeDTO } from '../types/NodeTypes';


/**
 * Thêm hoặc cập nhật các entities Node trên Cesium Viewer.
 * @param viewer Cesium Viewer instance
 * @param nodes Danh sách NodeDTO
 */
export const syncNodeEntities = (viewer: Cesium.Viewer, nodes: NodeDTO[]): void => {
    // Xóa tất cả entities hiện tại để đồng bộ hóa
    viewer.entities.removeAll(); 

    nodes.forEach(node => {
        const color = node.isHealthy ? Cesium.Color.GREEN : Cesium.Color.RED;
        
        viewer.entities.add({
            id: node.nodeId,
            position: Cesium.Cartesian3.fromDegrees(
                node.position.longitude,
                node.position.latitude,
                node.position.altitude * 1000 // Cesium dùng mét, ta chuyển Km sang mét
            ),
            point: {
                pixelSize: 12,
                color: color,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 3,
            },
            // Lưu dữ liệu Node vào properties để truy xuất khi click
            properties: { nodeData: node } 
        });
    });
};

/**
 * Cấu hình handler xử lý sự kiện click trên bản đồ.
 * @param viewer Cesium Viewer instance
 * @param setSelectedNode Action cập nhật trạng thái Node được chọn
 */
export const setupMapClickHandler = (viewer: Cesium.Viewer, setSelectedNode: (node: NodeDTO | null) => void): Cesium.ScreenSpaceEventHandler => {
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    handler.setInputAction((click : Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const picked = viewer.scene.pick(click.position);
        
        if (Cesium.defined(picked) && Cesium.defined(picked.id) && picked.id.properties && picked.id.properties.nodeData) {
            // Lấy dữ liệu NodeDTO từ entity
            const nodeData: NodeDTO = picked.id.properties.nodeData.getValue();
            setSelectedNode(nodeData);
        } else {
            // Bấm vào không gian trống
            setSelectedNode(null); 
        }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    
    return handler;
};
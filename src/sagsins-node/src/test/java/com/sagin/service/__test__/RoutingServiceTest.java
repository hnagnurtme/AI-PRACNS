package com.sagin.service.__test__;


import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.sagin.routing.IRoutingService;
import com.sagin.routing.RouteInfo;
import com.sagin.routing.RoutingService;
import com.sagin.routing.RoutingTable;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit Test cho lớp RoutingService.
 * Lớp này không cần mock vì nó hoạt động như một "trình quản lý" (manager)
 * cho các đối tượng RoutingTable (thật).
 */
class RoutingServiceTest {

    private IRoutingService routingService;

    // Các tuyến đường mẫu
    private RouteInfo route_N1_to_N3_via_N2; // Tốt
    private RouteInfo route_N1_to_N3_via_N4; // Xấu
    private RouteInfo route_N5_to_N3_via_N6;

    @BeforeEach
    void setUp() {
        // Khởi tạo một service mới cho mỗi test để đảm bảo sự cô lập
        routingService = new RoutingService();

        // Định nghĩa các tuyến đường mẫu
        // Tuyến TỐT: cost thấp, latency thấp (các trường khác dùng giá trị mặc định hoặc null)
        route_N1_to_N3_via_N2 = new RouteInfo();
        route_N1_to_N3_via_N2.setSourceNodeId("N1"); // Thêm nguồn
        route_N1_to_N3_via_N2.setDestinationNodeId("N3");
        route_N1_to_N3_via_N2.setNextHopNodeId("N2");
        route_N1_to_N3_via_N2.setPathNodeIds(List.of("N1", "N2", "N3")); // Thêm đường đi
        route_N1_to_N3_via_N2.setTotalCost(10.0);       // Giá trị quan trọng cho test
        route_N1_to_N3_via_N2.setTotalLatencyMs(5.0);   // Giá trị quan trọng cho test
        // (Giả sử các trường khác như băng thông, mất gói, độ tin cậy dùng mặc định
        //  để việc so sánh điểm trong routeScore hoạt động đúng)

        // Tuyến XẤU: cost cao, latency cao
        route_N1_to_N3_via_N4 = new RouteInfo();
        route_N1_to_N3_via_N4.setSourceNodeId("N1");
        route_N1_to_N3_via_N4.setDestinationNodeId("N3");
        route_N1_to_N3_via_N4.setNextHopNodeId("N4");
        route_N1_to_N3_via_N4.setPathNodeIds(List.of("N1", "N4", "N3"));
        route_N1_to_N3_via_N4.setTotalCost(100.0);      // Giá trị quan trọng cho test
        route_N1_to_N3_via_N4.setTotalLatencyMs(50.0);  // Giá trị quan trọng cho test

        // Tuyến cho node khác
        route_N5_to_N3_via_N6 = new RouteInfo();
        route_N5_to_N3_via_N6.setSourceNodeId("N5");
        route_N5_to_N3_via_N6.setDestinationNodeId("N3");
        route_N5_to_N3_via_N6.setNextHopNodeId("N6");
        route_N5_to_N3_via_N6.setPathNodeIds(List.of("N5", "N6", "N3"));
    }

    @Test
    @DisplayName("Test getBestRoute trả về đúng tuyến đường tốt nhất")
    void testGetBestRoute_SelectsBestRoute() {
        // === 1. SẮP XẾP ===
        // Thêm 2 tuyến đường (1 tốt, 1 xấu) vào bảng của N1
        routingService.updateRoute("N1", route_N1_to_N3_via_N2); // Tốt
        routingService.updateRoute("N1", route_N1_to_N3_via_N4); // Xấu
        
        // Thêm 1 tuyến đường cho N5 (để đảm bảo chúng không bị lẫn)
        routingService.updateRoute("N5", route_N5_to_N3_via_N6);

        // === 2. HÀNH ĐỘNG ===
        // Hỏi đường đi tốt nhất TỪ N1 ĐẾN N3
        RouteInfo bestRoute = routingService.getBestRoute("N1", "N3");

        // === 3. KHẲNG ĐỊNH ===
        assertNotNull(bestRoute);
        // Khẳng định rằng nó đã chọn đúng tuyến đường "TỐT" (via N2)
        assertEquals("N2", bestRoute.getNextHopNodeId());
    }

    @Test
    @DisplayName("Test getBestRoute trả về null khi không có đường đi")
    void testGetBestRoute_ReturnsNullForNoRoute() {
        // === 1. SẮP XẾP ===
        // Thêm tuyến đường cho N1
        routingService.updateRoute("N1", route_N1_to_N3_via_N2);

        // === 2. HÀNH ĐỘNG ===
        // Hỏi đường đi từ N2 (chưa có bảng định tuyến)
        RouteInfo routeFromN2 = routingService.getBestRoute("N2", "N3");
        
        // Hỏi đường đi từ N1 đến một đích không tồn tại
        RouteInfo routeToN99 = routingService.getBestRoute("N1", "N99");

        // === 3. KHẲNG ĐỊNH ===
        assertNull(routeFromN2);
        assertNull(routeToN99);
    }

    @Test
    @DisplayName("Test getBestRoute trả về null khi tham số là null")
    void testGetBestRoute_HandlesNullArguments() {
        // === HÀNH ĐỘNG & KHẲNG ĐỊNH ===
        assertNull(routingService.getBestRoute(null, "N3"));
        assertNull(routingService.getBestRoute("N1", null));
        assertNull(routingService.getBestRoute(null, null));
    }

    @Test
    @DisplayName("Test getRoutingTableForNode tạo bảng mới và dùng lại")
    void testGetRoutingTableForNode_CreatesAndReusesTable() {
        // === HÀNH ĐỘNG ===
        // Lấy bảng cho N1 lần đầu tiên
        RoutingTable table1 = routingService.getRoutingTableForNode("N1");

        // Thêm một tuyến đường vào N1
        routingService.updateRoute("N1", route_N1_to_N3_via_N2);

        // Lấy bảng cho N1 lần thứ hai
        RoutingTable table2 = routingService.getRoutingTableForNode("N1");

        // Lấy bảng cho N5 (bảng mới)
        RoutingTable table5 = routingService.getRoutingTableForNode("N5");

        // === KHẲNG ĐỊNH ===
        assertNotNull(table1); // Khẳng định bảng đã được tạo
        assertNotNull(table2);
        assertNotNull(table5);

        // Khẳng định table1 và table2 là CÙNG MỘT ĐỐI TƯỢNG (instance)
        assertSame(table1, table2, "Phải trả về cùng một đối tượng (instance) cho cùng một NodeId");
        
        // Khẳng định table1 và table5 là KHÁC NHAU
        assertNotSame(table1, table5, "Phải trả về các đối tượng (instance) khác nhau cho các NodeId khác nhau");

        // Khẳng định tuyến đường đã thêm vào table1 cũng tồn tại trong table2
        assertEquals(1, table2.getActiveRoutes().size());
        assertEquals(0, table5.getActiveRoutes().size()); // Bảng của N5 phải trống
    }

    @Test
    @DisplayName("Test updateRoute hoạt động chính xác")
    void testUpdateRoute_AddsRouteToCorrectTable() {
        // === HÀNH ĐỘNG ===
        routingService.updateRoute("N1", route_N1_to_N3_via_N2);
        routingService.updateRoute("N5", route_N5_to_N3_via_N6);

        // === KHẲNG ĐỊNH ===
        // Lấy bảng của N1 và kiểm tra
        RoutingTable tableN1 = routingService.getRoutingTableForNode("N1");
        assertNotNull(tableN1.getBestRoute("N3"));
        assertEquals("N2", tableN1.getBestRoute("N3").getNextHopNodeId());

        // Lấy bảng của N5 và kiểm tra
        RoutingTable tableN5 = routingService.getRoutingTableForNode("N5");
        assertNotNull(tableN5.getBestRoute("N3"));
        assertEquals("N6", tableN5.getBestRoute("N3").getNextHopNodeId());
    }
}
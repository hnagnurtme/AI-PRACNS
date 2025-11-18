import copy

class MockDBConnector:
    """
    Giả lập Database với 10 Nodes cố định để test thuật toán RL.
    Topology:
    - GS_HANOI (Node 0): Kết nối với SAT_1, SAT_2
    - GS_HCM (Node 9): Kết nối với SAT_7, SAT_8
    - SAT_1 đến SAT_8: Là các vệ tinh trung gian nối tiếp nhau.
    """

    def __init__(self):
        self.nodes = {}
        self._initialize_nodes()

    def _initialize_nodes(self):
        # --- 1. Template chuẩn cho 1 Node ---
        base_node_template = {
            "nodeId": "",
            "nodeName": "",
            "nodeType": "SATELLITE", # hoặc GROUND_STATION
            "position": {"latitude": 0.0, "longitude": 0.0, "altitude": 550.0},
            "velocity": {"velocityX": 7.6, "velocityY": 0.0, "velocityZ": 0.0},
            "communication": {
                "frequencyGHz": 26.5,       # Ka-band
                "bandwidthMHz": 1000,
                "transmitPowerDbW": 20,
                "antennaGainDb": 30,
                "beamWidthDeg": 1.5,
                "maxRangeKm": 5000.0,
                "minElevationDeg": 10,
                "ipAddress": "10.0.0.1",
                "port": 8080,
                "protocol": "UDP"
            },
            "isOperational": True,
            "batteryChargePercent": 100.0,
            "nodeProcessingDelayMs": 5.0,
            "packetLossRate": 0.001,      # 0.1% loss
            "resourceUtilization": 0.2,   # 20% CPU load
            "packetBufferCapacity": 5000,
            "currentPacketCount": 100,
            "weather": "CLEAR",
            "healthy": True,
            "neighbors": []
        }

        # --- 2. Định nghĩa 10 Nodes cụ thể ---
        
        # Node 0: Trạm mặt đất Hà Nội
        n0 = copy.deepcopy(base_node_template)
        n0.update({
            "nodeId": "GS_HANOI", "nodeName": "Ground Station Hanoi", "nodeType": "GROUND_STATION",
            "position": {"latitude": 21.0285, "longitude": 105.8542, "altitude": 0.0},
            "neighbors": ["SAT_1", "SAT_2"] # Start point
        })

        # Node 1: Vệ tinh ngay trên miền Bắc
        n1 = copy.deepcopy(base_node_template)
        n1.update({
            "nodeId": "SAT_1", "nodeName": "Vinasat-Like-1",
            "position": {"latitude": 19.0, "longitude": 106.0, "altitude": 600.0},
            "neighbors": ["GS_HANOI", "SAT_2", "SAT_3"]
        })

        # Node 2: Vệ tinh lệch Đông Bắc
        n2 = copy.deepcopy(base_node_template)
        n2.update({
            "nodeId": "SAT_2", "nodeName": "Starlink-A1",
            "position": {"latitude": 20.0, "longitude": 108.0, "altitude": 550.0},
            "neighbors": ["GS_HANOI", "SAT_1", "SAT_4"]
        })

        # Node 3: Vệ tinh miền Trung (Đà Nẵng)
        n3 = copy.deepcopy(base_node_template)
        n3.update({
            "nodeId": "SAT_3", "nodeName": "Vinasat-Like-2",
            "position": {"latitude": 16.0, "longitude": 107.0, "altitude": 600.0},
            "neighbors": ["SAT_1", "SAT_4", "SAT_5"]
        })

        # Node 4: Vệ tinh ngoài biển Đông
        n4 = copy.deepcopy(base_node_template)
        n4.update({
            "nodeId": "SAT_4", "nodeName": "Starlink-A2",
            "position": {"latitude": 16.5, "longitude": 110.0, "altitude": 550.0},
            "neighbors": ["SAT_2", "SAT_3", "SAT_6"]
        })

        # Node 5: Vệ tinh Nam Trung Bộ (Nha Trang)
        n5 = copy.deepcopy(base_node_template)
        n5.update({
            "nodeId": "SAT_5", "nodeName": "Vinasat-Like-3",
            "position": {"latitude": 12.0, "longitude": 108.0, "altitude": 600.0},
            "neighbors": ["SAT_3", "SAT_6", "SAT_7"]
        })

        # Node 6: Vệ tinh Trường Sa
        n6 = copy.deepcopy(base_node_template)
        n6.update({
            "nodeId": "SAT_6", "nodeName": "Starlink-A3",
            "position": {"latitude": 11.0, "longitude": 112.0, "altitude": 550.0},
            "neighbors": ["SAT_4", "SAT_5", "SAT_8"]
        })

        # Node 7: Vệ tinh Đông Nam Bộ
        n7 = copy.deepcopy(base_node_template)
        n7.update({
            "nodeId": "SAT_7", "nodeName": "Vinasat-Like-4",
            "position": {"latitude": 10.5, "longitude": 107.0, "altitude": 600.0},
            "neighbors": ["SAT_5", "GS_HCM", "SAT_8"]
        })

        # Node 8: Vệ tinh Cà Mau / Biển Tây
        n8 = copy.deepcopy(base_node_template)
        n8.update({
            "nodeId": "SAT_8", "nodeName": "Starlink-A4",
            "position": {"latitude": 9.0, "longitude": 104.0, "altitude": 550.0},
            "neighbors": ["SAT_6", "SAT_7", "GS_HCM"]
        })

        # Node 9: Trạm mặt đất HCM
        n9 = copy.deepcopy(base_node_template)
        n9.update({
            "nodeId": "GS_HCM", "nodeName": "Ground Station HCM", "nodeType": "GROUND_STATION",
            "position": {"latitude": 10.8231, "longitude": 106.6297, "altitude": 0.0},
            "neighbors": ["SAT_7", "SAT_8"] # End point
        })

        # Lưu vào dict
        self.nodes = {
            "GS_HANOI": n0, "SAT_1": n1, "SAT_2": n2, "SAT_3": n3, "SAT_4": n4,
            "SAT_5": n5, "SAT_6": n6, "SAT_7": n7, "SAT_8": n8, "GS_HCM": n9
        }

    # --- Interface Methods (Required by StateBuilder) ---

    def get_node(self, node_id: str, projection=None):
        """Trả về thông tin của 1 node."""
        # Mô phỏng query database, trả về bản copy để tránh side-effect
        node = self.nodes.get(node_id)
        if node:
            return copy.deepcopy(node)
        return None

    def get_nodes(self, node_ids: list):
        """Trả về danh sách nodes."""
        result = []
        for nid in node_ids:
            n = self.get_node(nid)
            if n:
                result.append(n)
        return result

    def get_all_nodes(self, projection=None):
        """Trả về tất cả 10 node (dùng cho hàm generate_packet)."""
        return [copy.deepcopy(n) for n in self.nodes.values()]
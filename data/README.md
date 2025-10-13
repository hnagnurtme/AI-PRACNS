# Data Directory

CREATE MONGODB DATABASE

```
{
  "_id": ObjectId,                     // ID nội bộ MongoDB, tự động sinh

  "nodeId": "LEO-001",                 // Mã định danh duy nhất của node (ví dụ: "LEO-001", "GS-03")
  "nodeName": "Sat-LEO-1",             // Tên hiển thị hoặc nhãn dễ hiểu

  "type": "LEO_SATELLITE",             // Enum: GROUND_STATION | LEO_SATELLITE | MEO_SATELLITE | GEO_SATELLITE
                                       // Xác định loại node: trạm mặt đất hay vệ tinh tầng nào

  // --- TỌA ĐỘ KHÔNG GIAN ---
  "position": {                        // Vị trí không gian 3D trong hệ toạ độ địa tâm
    "latitude": 10.7626,               // Vĩ độ (°), Bắc (+) / Nam (-)
    "longitude": 106.6602,             // Kinh độ (°), Đông (+) / Tây (-)
    "altitude": 550.0                  // Độ cao (km) so với mực nước biển hoặc tâm Trái Đất (tùy hệ quy chiếu)
  },

  // --- ĐỘNG HỌC ---
  "velocity": {                        // Vector vận tốc (km/s) của node trong hệ toạ độ quỹ đạo
    "velocityX": 0.0,                  // Thành phần vận tốc theo trục X (Đông–Tây)
    "velocityY": 7.56,                 // Thành phần vận tốc theo trục Y (Bắc–Nam)
    "velocityZ": 0.0                   // Thành phần vận tốc theo trục Z (Lên–Xuống)
  },

  // --- QUỸ ĐẠO (nếu là vệ tinh) ---
  "orbit": {                           // Thông tin quỹ đạo (null nếu là trạm mặt đất)
    "semiMajorAxisKm": 6871.0,         // Bán trục lớn của quỹ đạo (km)
    "eccentricity": 0.001,             // Độ lệch tâm: 0 = tròn, >0 = elip
    "inclinationDeg": 53.0,            // Góc nghiêng quỹ đạo so với mặt phẳng xích đạo (°)
    "raanDeg": 45.0,                   // Right Ascension of Ascending Node (RAAN) – hướng quỹ đạo trong không gian
    "argumentOfPerigeeDeg": 0.0,       // Góc cận điểm – xác định hướng elip
    "trueAnomalyDeg": 12.5              // Vị trí tức thời của vệ tinh trên quỹ đạo (°)
  },

  // --- THÔNG SỐ TRUYỀN THÔNG ---
  "communication": {
    "frequencyGHz": 14.25,             // Tần số hoạt động (GHz), ví dụ: 14.25 GHz (Ku-band)
    "bandwidthMHz": 500.0,             // Băng thông khả dụng (MHz)
    "transmitPowerDbW": 20.0,          // Công suất phát (dBW)
    "antennaGainDb": 15.0,             // Độ lợi anten (dB)
    "beamWidthDeg": 3.0,               // Độ rộng chùm sóng – Field of View (°)
    "maxRangeKm": 2000.0,              // Tầm liên lạc tối đa (km)
    "minElevationDeg": 10.0            // Góc nâng tối thiểu để đảm bảo liên lạc (°)
    "ipAddress": "10.0.0.12",
    "port": 8080,
  },

  // --- TRẠNG THÁI HOẠT ĐỘNG ---
  "status": {
    "active": true,                    // Node đang hoạt động hay tạm dừng
    "batteryChargePercent": 82.5,      // (tuỳ chọn) phần trăm pin hiện tại, đặc biệt hữu ích cho vệ tinh nhỏ
    "lastUpdated": "2025-10-13T16:00:00Z" // Thời điểm cập nhật trạng thái gần nhất (UTC)
  },

  // --- THÔNG TIN BỔ TRỢ ---
  "metadata": {
    "operator": "VNPT Space",          // Tổ chức hoặc quốc gia vận hành node
    "launchDate": "2024-03-01T00:00:00Z", // Ngày phóng hoặc triển khai node
    "notes": "Experimental low-latency link node" // Ghi chú thêm, dùng tự do
  }
}
```

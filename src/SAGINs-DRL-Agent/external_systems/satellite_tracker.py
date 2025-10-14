# external_systems/satellite_tracker.py
import time
import random
from pymongo import MongoClient

class SatelliteTracker:
    """Hệ thống cập nhật vị trí vệ tinh real-time vào MongoDB"""
    
    def __init__(self):
        # Kết nối MongoDB với authentication
        host: str = "localhost"
        port: int = 27017
        username: str = "user"
        password: str = "password123"
        auth_source: str = "admin"
        connection_string = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_source}"
        self.client = MongoClient(connection_string)
        
        # Test connection
        try:
            self.client.admin.command('ping')
            print("✅ Connected with authentication to MongoDB")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
        
        self.db = self.client['SAGSINS']
        self.nodes = self.db['network_nodes']
    
    def update_satellite_positions(self):
        """Cập nhật vị trí vệ tinh real-time"""
        satellites = self.nodes.find({"nodeType": {"$in": ["LEO_SATELLITE", "MEO_SATELLITE", "GEO_SATELLITE"]}})
        
        for sat in satellites:
            # Tính toán vị trí mới dựa trên orbital mechanics
            new_position = self._calculate_new_position(sat)
            
            # UPDATE MONGODB REAL-TIME
            self.nodes.update_one(
                {"_id": sat["_id"]},
                {"$set": {
                    "position": new_position,
                    "lastUpdated": int(time.time() * 1000)
                }}
            )
            print(f"📍 Updated {sat['_id']} position")
    
    def _calculate_new_position(self, satellite):
        """Tính vị trí mới từ quỹ đạo"""
        orbit = satellite.get('orbit', {})
        # ... orbital calculations ...
        return {
            "latitude": random.uniform(-90, 90),
            "longitude": random.uniform(-180, 180),
            "altitude": orbit.get('semiMajorAxisKm', 6800) - 6371
        }

# Chạy trong background
tracker = SatelliteTracker()
while True:
    tracker.update_satellite_positions()
    time.sleep(60)  # Update mỗi phút
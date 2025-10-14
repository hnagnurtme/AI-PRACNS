# external_systems/weather_service.py
import time
import random
from pymongo import MongoClient

class WeatherService:
    """Service cập nhật thời tiết real-time vào MongoDB"""
    
    WEATHER_CONDITIONS = ['CLEAR', 'LIGHT_RAIN', 'MODERATE_RAIN', 'HEAVY_RAIN', 'SEVERE_STORM']
    
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
    
    def update_weather_conditions(self):
        """Cập nhật thời tiết cho tất cả nodes"""
        all_nodes = self.nodes.find({})
        
        for node in all_nodes:
            if random.random() < 0.2:  # 20% chance thay đổi thời tiết
                new_weather = random.choice(self.WEATHER_CONDITIONS)
                
                # UPDATE MONGODB REAL-TIME
                self.nodes.update_one(
                    {"_id": node["_id"]},
                    {"$set": {
                        "weather": new_weather,
                        "lastUpdated": int(time.time() * 1000)
                    }}
                )
                print(f"🌤️ Updated {node['_id']} weather: {new_weather}")

# Chạy weather service
weather_service = WeatherService()
while True:
    weather_service.update_weather_conditions()
    time.sleep(300)  # Update mỗi 5 phút
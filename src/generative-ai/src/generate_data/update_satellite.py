import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import math
import requests
import time
from skyfield.api import load, EarthSatellite, wgs84
from pathlib import Path

# Tìm đường dẫn tới serviceAccountKey.json
HERE = Path(__file__).resolve().parent  # thư mục hiện tại: src/generate_data/
SERVICE_KEY_PATH = HERE.parent / "serviceAccountKey.json"  # lên 1 cấp về src/ rồi tìm file

# Cấu hình Firebase
if not SERVICE_KEY_PATH.exists():
    print(f"ERROR: Service account key not found at {SERVICE_KEY_PATH}")
    exit(1)

cred = credentials.Certificate(str(SERVICE_KEY_PATH))
firebase_admin.initialize_app(cred)
db = firestore.client()

R_EARTH = 6371.0  # Bán kính Trái Đất tính bằng km
SPEED_OF_LIGHT = 299792.458  # Tốc độ ánh sáng tính bằng km/s
PROCESSING_DELAY = 0.5  # Độ trễ xử lý tín hiệu vệ tinh tính bằng giây

def get_weather(lat: float, lon: float) -> dict:
    #  Lấy thời tiết từ Open-Meteo API
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,cloud_cover&timezone=auto"
        res = requests.get(url)
        data = res.json()
        # Lấy giá trị hiện tại (index 0)
        precipitation = data['hourly']['precipitation'][0] # mm / h
        cloud_cover = data['hourly']['cloud_cover'][0]  # %
        return {
            "precipitation_mm_h": precipitation,
            "cloud_cover_percent": cloud_cover
        }
    except Exception as e:
        print(f"Error fetching weather data for ({lat}, {lon}): {e}")
        return {
            "precipitation_mm_h": 0.0,
            "cloud_cover_percent": 0.0
        }


def calculate_attenuation_from_weather(precipitation: float, freq_ghz: float, path_length_km: float) -> float:
    # Tính attenation (dB) từ mưa theo ITU-R P.838-3
    try:
        gamma = 0.0001 # Hệ số cho UHF
        a, b, c = 0.67, 0.85, 1.0 # hệ số
        attenuation = gamma * (precipitation ** a) * (freq_ghz ** b) * (path_length_km ** c)
        return round(attenuation, 3)
    except Exception as e:
        print(f"Error calculating attenuation: {e}")
        return 0.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Tính khoảng cách bề mặt trái đất giữa hai điểm (lat1, lon1) và (lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH * c

# Tính khoảng cách đường truyền tín hiệu từ trạm mặt đất đến vệ tinh
def calculate_path_length_km(ground_lat: float, ground_lon: float, ground_alt_km: float,
                             sat_lat: float, sat_lon: float, sat_alt_km: float) -> float:
    surface_distance = haversine_distance(ground_lat, ground_lon, sat_lat, sat_lon)
    height_diff = abs(sat_alt_km - ground_alt_km)
    path_length = math.sqrt(surface_distance**2 + height_diff**2)
    return round(path_length, 3)

def update_satellite(node_id: str, tle_line1: str, tle_line2: str):
    # Cập nhật vị trí, vận tốc vệ tinh từ TLE 
    try:
        ts = load.timescale()
        t = ts.now()
        
        satellite = EarthSatellite(tle_line1, tle_line2, node_id, ts)
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        
        lat = subpoint.latitude.degrees
        lon = subpoint.longitude.degrees
        alt_km = subpoint.elevation.km # độ cao so với mực nước biển tính bằng km
        
        vx, vy, vz = geocentric.velocity.km_per_s
        
        db.collection("satellites").document(node_id).update({
            "position": {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt_km  
            },
            "velocity_km_s": {
                "vx": vx,
                "vy": vy,
                "vz": vz
            },
            "last_updated": datetime.now().timestamp() * 1000  # milliseconds
        })
        print(f"Updated satellite {node_id}: Pos({lat}, {lon}, {alt_km} km), Vel({vx}, {vy}, {vz} km/s)")
    except Exception as e:
        print(f"Error loading timescale: {node_id}: {e}")
    
def update_ground_sea_weather(node_id: str, lat: float, lon: float, collection: str):
    # Cập nhật thời tiết cho ground station hoặc sea station
    weather = get_weather(lat, lon)
    freq_ghz = 0.43 
    path_length_km = 1210.873
    attenuation_db = calculate_attenuation_from_weather(weather['precipitation_mm_h'], freq_ghz, path_length_km)
    snr_reduction_db = attenuation_db * 0.5
    
    db.collection(collection).document(node_id).update({
        "weather": {
            "precipitation_mm_h": weather["precipitation_mm_h"],
            "cloud_cover_percent": weather["cloud_cover_percent"],
            "attenuation_db": attenuation_db,
            "snr_reduction_db": snr_reduction_db
        },
        "last_updated": datetime.now().timestamp() * 1000  # milliseconds
    })
    print(f"Updated {collection} {node_id} weather: {weather}, Attenuation: {attenuation_db} dB")

def main():
    while True:
        # Lấy satellite từ Firestore
        satellites = [doc.to_dict() for doc in db.collection("satellites").stream()]
        for sat in satellites:
            update_satellite(sat["nodeId"], sat['orbit']['tle']['line1'], sat['orbit']['tle']['line2'])
            time.sleep(1)  # tránh gọi API quá nhanh
        
        # Lấy ground stations và sea stations từ Firestore
        ground_stations = [doc.to_dict() for doc in db.collection("ground_stations").stream()]
        sea_stations = [doc.to_dict() for doc in db.collection("sea_stations").stream()]
        
        for gs in ground_stations:
            update_ground_sea_weather(gs["nodeId"], gs["position"]["latitude"], gs["position"]["longitude"], "ground_stations")
            time.sleep(1)  # tránh gọi API quá nhanh
        for ss in sea_stations:
            update_ground_sea_weather(ss["nodeId"], ss["position"]["latitude"], ss["position"]["longitude"], "sea_stations")
            time.sleep(1)  # tránh gọi API quá nhanh
        
        time.sleep(240) # chờ 4 phút trước khi cập nhật lần tiếp theo

if __name__ == "__main__":
    main()
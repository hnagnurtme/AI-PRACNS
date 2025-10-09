import json 
from pathlib import Path
from jsonschema import validate
from datetime import datetime
import math
import requests
import sys

# locate project root (generative-ai) relative to this script
ROOT = Path(__file__).resolve().parents[2]  # .../generative-ai
HERE = ROOT
SCHEMA_PATH = HERE / "data" / "schema" / "ground.json"
RAW_PATH = HERE / "data" / "raw" / "sea_stations" / "sea_stations_raw_01.json"
SAT_DATA_PATH = HERE / "data" / "processed" / "satellites" / "satellites_processed_01.json"
OUT = HERE / "data" / "processed" / "sea_station" / "sea_station_processed.json"

if not SCHEMA_PATH.exists():
    print("Missing schema:", SCHEMA_PATH)
    sys.exit(1)
if not RAW_PATH.exists():
    print("Missing raw input:", RAW_PATH)
    sys.exit(1)

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
RAW = json.loads(RAW_PATH.read_text(encoding="utf-8"))
SAT_DATA = json.loads(SAT_DATA_PATH.read_text(encoding="utf-8"))

R_EARTH = 6371.0  # bán kính trái đất km
processed = []

def get_weather(lat: float, lng: float) -> dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=precipitation,cloud_cover&timezone=auto"
    res = requests.get(url)
    data = res.json()
    # Lấy giá trị hiện tại (index 0)
    precipitation = data['hourly']['precipitation'][0] # mm / h
    cloud_cover = data['hourly']['cloud_cover'][0]  # %
    return {
        "precipitation_mm_h": precipitation,
        "cloud_cover_percent": cloud_cover
    }

# Tính attenuation (dB) từ mưa theo ITU-R P.838
def calculate_attenuation_from_weather(precipitation: float, freq_ghz: float, path_length_km: float) -> float:
    gamma = 0.01 # hệ số cơ bản cho ku-band
    a, b, c = 0.67, 0.85, 1.0 # hệ số
    A = gamma * (precipitation ** a) * (freq_ghz ** b) * (path_length_km ** c)
    return A

# Tính khoảng cách bề mặt trái đất giữa trạm mặt đất và vệ tinh
def haversine_distance(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH * c

def calculate_path_length_km(ground_lat, ground_lon, ground_alt_km, sat_lat, sat_lon, sat_alt_km):
    surface_distance = haversine_distance(ground_lat, ground_lon, sat_lat, sat_lon)
    height_diff = abs(sat_alt_km - ground_alt_km)
    path_length = math.sqrt(surface_distance**2 + height_diff**2)
    return round(path_length, 3)

def preprocess_sea_station(station: dict, sat_data: list) -> dict:
    try:
        print(f"Start processing sea station {station.get('station_id', station.get('name','unknown'))}")
        node_id = f"sea_{station['station_id']}"
        altutude_km = 0.01 # giả sử trạm biển cao 10m so với mực nước biển
        min_horizon = 10
        
        maintenance_status = "operational"
        healthy = True
        link_available = True
        
        bandwidth_mbps = 2500.0
        freq_hz = 14000000000.0 # 14 GHz (Ku-band)
        
        coverage_radius_km = math.sqrt(2 * R_EARTH * altutude_km + altutude_km**2) * math.cos(math.radians(min_horizon))
        coverage_radius_km = round(coverage_radius_km, 3)
        
        # Lấy thời tiết hiện tại
        lat = float(station.get("latitude", 0.0))
        lon = float(station.get("longitude", 0.0))
        weather = get_weather(lat, lon)
        precipitation = weather.get("precipitation_mm_h", 0.0)
        cloud_cover = weather.get("cloud_cover_percent", 0.0)
        
        # Tìm vệ tinh gần nhất trong phạm vi phủ sóng
        path_length_km = 1210.873
        if sat_data:
            min_distance = float('inf')
            for sat in sat_data:
                dist = calculate_path_length_km(
                    lat, lon, altutude_km,
                    sat["position"]["latitude"], sat["position"]["longitude"], sat["position"]["altitude"]
                )
                if dist < min_distance:
                    min_distance = dist
            path_length_km = min_distance
            
        # Tính attenuation_dB và snrReduction_dB từ thời tiết
        attenuation_dB = calculate_attenuation_from_weather(precipitation, freq_hz / 1e9, path_length_km)
        snrReduction_dB = attenuation_dB * 0.5
        
        # Tính latencyMs từ path_length_km và cộng thêm 0.5ms cho xử lý
        latencyMs = (path_length_km / 300.0) * 1000.0 + 0.5  # ánh sáng trong chân không ~300 km/ms
        
        # Chuẩn hóa thời gian
        last_seen = "2019-03-18T07:10:37Z"
        last_updated_dt = datetime.strptime(last_seen, "%Y-%m-%dT%H:%M:%SZ")
        last_updated = int(last_updated_dt.timestamp() * 1000)
        
        item = {
            "nodeId": node_id,
            "nodeName": station.get("name", "unknown"),
            "nodeType": "SEA",
            "position": {
                "latitude": lat,
                "longitude": lon,
                "altitude": altutude_km
            },
            "velocity": {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0
            },
            "bandwidth": bandwidth_mbps,
            "capacityMbps": bandwidth_mbps * 1.5,
            "coverageRadiusKm": coverage_radius_km,
            "maintenanceStatus": maintenance_status,
            "healthy": healthy,
            "linkAvailable": link_available,
            "latencyMs": latencyMs, # độ trễ từ trạm biển đến vệ tinh gần nhất (đơn vị ms)
            "lastUpdated": last_updated_dt.isoformat() + "Z",
            "pathLengthKm": path_length_km, # khoảng cách đến vệ tinh gần nhất
            "weather": {
                "precipitation_mm_h": precipitation,
                "cloud_cover_percent": cloud_cover,
                "attenuation_dB": attenuation_dB,
                "snrReduction_dB": snrReduction_dB
            },
            "meta": {
                "description": station.get("description", ""),
                "owner": station.get("owner", "unknown"),
                "hull": station.get("hull", ""),
                "location_raw": station.get("location_raw", ""),
                "note": station.get("note", "")
            }
        }
        
        validate(instance=item, schema=SCHEMA)
        return item
    except Exception as e:
        print(f"Error processing sea station {station.get('station_id', station.get('name','unknown'))}: {e}")
        return None

def main():
    for station in RAW:
        item = preprocess_sea_station(station, SAT_DATA)
        if item:
            processed.append(item)
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(processed, indent=2), encoding="utf-8")
    print(f"Đã xử lý và lưu dữ liệu trạm biển, có {len(processed)} trạm ->", OUT)

if __name__ == "__main__":
    main()
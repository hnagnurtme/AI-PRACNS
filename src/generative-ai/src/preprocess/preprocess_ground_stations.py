import json 
from pathlib import Path
from jsonschema import validate
from datetime import datetime
import sys
import math
import re
import requests

# locate project root (generative-ai) relative to this script
ROOT = Path(__file__).resolve().parents[2]  # .../generative-ai
HERE = ROOT
SCHEMA_PATH = HERE / "data" / "schema" / "ground.json"
RAW_PATH = HERE / "data" / "raw" / "ground_station" / "ground_station_raw.json"
SAT_DATA_PATH = HERE / "data" / "processed" / "satellites" / "satellites_processed_01.json"
OUT = HERE / "data" / "processed" / "ground_station" / "ground_station_processed.json"

if not SCHEMA_PATH.exists():
    print("Missing schema:", SCHEMA_PATH)
    sys.exit(1)
if not RAW_PATH.exists():
    print("Missing raw input:", RAW_PATH)
    sys.exit(1)

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
RAW = json.loads(RAW_PATH.read_text(encoding="utf-8"))
SAT_DATA = json.loads(SAT_DATA_PATH.read_text(encoding="utf-8"))

processed = []
R_EARTH = 6371.0  # bán kính trái đất km

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

def calculate_coverage_radius(altitude_km: float, min_horizon_deg: float) -> float:
    try:
        h = altitude_km
        min_horizon_rad = math.radians(min_horizon_deg)
        radius = math.sqrt(2 * R_EARTH * h + h**2) * math.cos(min_horizon_rad)
        return round(radius, 3)
    except Exception as e:
        print(f"Error calculating coverage radius for altitude {altitude_km} km and min_horizon {min_horizon_deg}°: {e}")
        return 500.0

def preprocess_ground_station(station: dict, sat_data: list) -> dict:
    try:
        node_id = f"gs_{station.get('id', station.get('name', 'unknown'))}"
        altitude = float(station.get("altitude", 0)) / 1000.0  # m -> km

        status = str(station.get("status", "Offline")).lower()
        if status == "online":
            maintenance_status = "operational"
            healthy = True
            link_available = True
        else:
            maintenance_status = "maintenance"
            healthy = False
            link_available = False
        
        bandwidth_mbps = 2500.0

        antenna_field = station.get("antenna")
        antenna_obj = None
        if isinstance(antenna_field, dict):
            antenna_obj = antenna_field
        elif isinstance(antenna_field, list) and len(antenna_field) > 0:
            antenna_obj = antenna_field[0] if isinstance(antenna_field[0], dict) else None

        # Điều chỉnh bandwidth dựa trên tần số
        freq_hz = 0
        if antenna_obj:
            freq_hz = antenna_obj.get("frequency", 0) or 0
            try:
                freq_mhz = float(freq_hz) / 1e6
            except Exception:
                freq_mhz = 0
            if 400 <= freq_mhz <= 470:
                bandwidth_mbps = 2500.0
            elif freq_mhz > 1000:
                bandwidth_mbps = 10000.0
        
        # Tính coverageRadiusKm dựa trên altitude (km) và min_horizon (độ)
        coverage_radius_km = calculate_coverage_radius(altitude, station.get("min_horizon", 10))

        # normalize lastSeen / created_at / timestamps to ISO8601 string (schema expects string date-time)
        last_seen = station.get("last_seen") or station.get("created_at") or station.get("lastSeen") or station.get("updated_at")
        if last_seen is None:
            last_updated_str = datetime.utcnow().isoformat() + "Z"
        elif isinstance(last_seen, (int, float)):
            # assume milliseconds epoch
            ts = float(last_seen) / 1000.0
            last_updated_str = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        elif isinstance(last_seen, str):
            s = last_seen.strip()
            # numeric string epoch (seconds or milliseconds)
            if re.fullmatch(r"\d{10,13}", s):
                if len(s) >= 13:
                    ts = int(s) / 1000.0
                else:
                    ts = int(s)
                last_updated_str = datetime.utcfromtimestamp(ts).isoformat() + "Z"
            else:
                # try ISO parse, keep offset if present
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    # produce RFC3339-like string (keep timezone info if present)
                    last_updated_str = dt.isoformat()
                except Exception:
                    last_updated_str = datetime.utcnow().isoformat() + "Z"
        else:
            last_updated_str = datetime.utcnow().isoformat() + "Z"
        
        # Lấy dữ liệu thời tiết
        weather = get_weather(float(station.get("lat", 0.0)), float(station.get("lng", 0.0)))
        precipitation = weather.get("precipitation_mm_h", 0.0)
        cloud_cover = weather.get("cloud_cover_percent", 0.0)

        path_length_km = 400.0
        if sat_data:
            # Tìm vệ tinh gần nhất
            min_distance = float('inf')
            for sat in sat_data:
                surface_dist = haversine_distance(
                    station.get("lat", 0.0), station.get("lng", 0.0),
                    sat["position"]["latitude"], sat["position"]["longitude"]
                )
                height_diff = abs(sat["position"]["altitude"] - altitude)
                distance = math.sqrt(surface_dist**2 + height_diff**2)
                if distance < min_distance:
                    min_distance = distance
            path_length_km = round(min_distance, 3)
        
        # Tính attenuation dB từ mưa
        attenuation_dB = calculate_attenuation_from_weather(precipitation, freq_hz / 1e9, path_length_km)
        
        # Tính độ trễ dựa trên khoảng cách từ trạm mặt đất đến vệ tinh gần nhất (path_length_km)
        latencyMs = (path_length_km / 300.0) * 1000.0 + 0.5  # ánh sáng trong chân không ~300 km/ms
        
        item = {
            "nodeId": node_id,
            "nodeName": station.get("name", "unknown"),
            "nodeType": "GROUND_STATION",
            "position": {
                "latitude": float(station.get("lat", 0.0)),
                "longitude": float(station.get("lng", 0.0)),
                "altitude": altitude
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
            "latencyMs": latencyMs, # độ trễ từ trạm mặt đất đến vệ tinh gần nhất (đơn vị s)
            "lastUpdated": last_updated_str,
            "pathLengthKm": path_length_km, # khoảng cách đến vệ tinh gần nhất
            "meta": {
                "description": station.get("description", ""),
                "owner": station.get("owner", "unknown"),
                "qthlocator": station.get("qthlocator", ""),
                "antenna": antenna_field or []
            },
            "weather": {
                "precipitation_mm_h": precipitation,
                "cloud_cover_percent": cloud_cover,
                "attenuation_dB": attenuation_dB,
                "snrReduction_dB": round(attenuation_dB * 0.5, 3)  # giả sử SNR giảm một nửa attenuation
            }
        }

        validate(instance=item, schema=SCHEMA)
        return item
    except Exception as e:
        print(f"Error processing ground station {station.get('id', station.get('name','unknown'))}: {e}")
        return None
def main():
    for ground_station in RAW:
        item = preprocess_ground_station(ground_station, SAT_DATA)
        if item:
            processed.append(item)
    
    # lưu kết quả
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(processed, indent=2), encoding="utf-8")
    print(f"Đã xử lý và lưu dữ liệu trạm mặt đất, có {len(processed)} trạm ->", OUT)

if __name__ == "__main__":
    main()
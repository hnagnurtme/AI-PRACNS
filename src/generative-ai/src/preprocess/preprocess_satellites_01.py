import json 
from pathlib import Path
from jsonschema import validate
import math
import time
from datetime import datetime
from skyfield.api import load, EarthSatellite, wgs84
import sys

# locate project root (generative-ai) relative to this script
import sys
ROOT = Path(__file__).resolve().parents[2]  # .../generative-ai
HERE = ROOT
SCHEMA_PATH = HERE / "data" / "schema" / "satellite.json"
RAW_PATH = HERE / "data" / "raw" / "satellites" / "satellites_raw_01.json"
OUT = HERE / "data" / "processed" / "satellites" / "satellites_processed_01.json"

if not SCHEMA_PATH.exists():
    print("Missing schema:", SCHEMA_PATH)
    sys.exit(1)
if not RAW_PATH.exists():
    print("Missing raw input:", RAW_PATH)
    sys.exit(1)

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
RAW = json.loads(RAW_PATH.read_text(encoding="utf-8"))

# const 
MU = 398600.4418  # km^3/s^2, hằng số hấp dẫn tiêu chuẩn của Trái Đất
R_EARTH = 6371.0  # km, bán kính Trái Đ

def calculate_position_from_tle(tle_line1: str, tle_line2: str, epoch: str, mean_motion: float | None = None) -> dict:
    # Tính toán vị trí (lat/lon/alt) từ TLE sử dụng thư viện skyfield
    try:
        ts = load.timescale()
        epoch_dt = datetime.strptime(epoch, "%Y-%m-%dT%H:%M:%S.%f")
        t = ts.utc(epoch_dt.year, epoch_dt.month, epoch_dt.day, epoch_dt.hour, epoch_dt.minute,
                   epoch_dt.second + epoch_dt.microsecond / 1e6)
        
        # Tạo đối tượng vệ tinh từ TLE (truyền timescale để tránh một số lỗi)
        satellite = EarthSatellite(tle_line1, tle_line2, "SATELLITE", ts)
        
        # Tính vị trí địa lý
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        altitude = subpoint.elevation.km  # km
        print(f"Calculated position: lat={latitude}, lon={longitude}, alt={altitude} km") # debug
        
        return {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude
        }
    except Exception as e:
        print(f"Error calculating position from TLE: {e}")
        # fallback: nếu có mean_motion, tính semi-major axis và altitude (km)
        if mean_motion:
            n_rad_s = float(mean_motion) * 2 * math.pi / 86400.0  # rad/s
            semi_major_axis_km = (MU / (n_rad_s ** 2)) ** (1 / 3)
            altitude_km = semi_major_axis_km - R_EARTH  # km
        else:
            altitude_km = None
        return {"latitude": 0.0, "longitude": 0.0, "altitude": altitude_km}

def calculate_velocity_from_tle(tle_line1: str, tle_line2: str, epoch: str) -> dict:
    try:
        ts = load.timescale()
        epoch_dt = datetime.strptime(epoch, "%Y-%m-%dT%H:%M:%S.%f")
        t = ts.utc(epoch_dt.year, epoch_dt.month, epoch_dt.day, epoch_dt.hour, epoch_dt.minute,
                   epoch_dt.second + epoch_dt.microsecond / 1e6)
        
        satellite = EarthSatellite(tle_line1, tle_line2, "SATELLITE", ts)
        geocentric = satellite.at(t)
        
        vx, vy, vz = geocentric.velocity.km_per_s  # km/s
        
        return {
            "vx": vx,
            "vy": vy,
            "vz": vz
        }
    except Exception as e:
        print(f"Error calculating velocity from TLE: {e}")
        return {
            "vx": 0.0, 
            "vy": 0.0, 
            "vz": 0.0
        }
        
def preprocess_satellite(sat: dict) -> dict:
    # Chuẩn hóa dữ liệu vệ tinh theo schema
    try:
        # 1. Tính toán chu kỳ quỹ đạo từi mean motion
        mean_motion = float(sat["MEAN_MOTION"])
        period_min = 1440.0 / mean_motion if mean_motion > 0 else 0.0
        
        # 2. Tính toán bán trục lớn (semi-major axis) từ mean motion
        n_rad_s = mean_motion * 2 * math.pi / 86400.0  # rad/s
        semi_major_axis_km = (MU / (n_rad_s ** 2)) ** (1 / 3)
        semi_major_axis_m = semi_major_axis_km * 1000.0
        
        # 3. Xác định loại quỹ đạo 
        eccentricity = float(sat["ECCENTRICITY"])
        if eccentricity < 0 or eccentricity >= 1:
            orbit_type = "UNKNOWN"
        else:
            if period_min < 128:
                orbit_type = "LEO"  # Low Earth Orbit
            elif period_min >= 128 and period_min < 1436:
                orbit_type = "MEO"  # Medium Earth Orbit
            elif abs(period_min - 1436) < 10:
                orbit_type = "GEO"  # Geostationary Orbit
            else:
                orbit_type = "OTHER"
        
        # 4. Tính toán vị trí từ TLE
        position = calculate_position_from_tle(
            sat["TLE"]["line1"],
            sat["TLE"]["line2"],
            sat["EPOCH"],
            mean_motion=mean_motion
        )
        
        # 5. Tính toán vận tốc từ TLE
        velocity = calculate_velocity_from_tle(
            sat["TLE"]["line1"],
            sat["TLE"]["line2"],
            sat["EPOCH"]
        )
        
        # 5. Xác định payloadType
        payload_type = "transparent"  # Mặc định
        if "STARLINK" in sat["OBJECT_NAME"].upper():
            payload_type = "regenerative"

        # 6. Tạo object chuẩn hóa 
        item = {
            "nodeId": sat["OBJECT_ID"],
            "nodeName": sat["OBJECT_NAME"],            
            "nodeType": "SATELLITE",
            "position": position, # lat/lon/alt (km)
            "orbit": {
                "type": orbit_type,
                "inclination": float(sat["INCLINATION"]),  # độ nghiêng
                "period": period_min, 
                "semiMajorAxis": semi_major_axis_m,  # m
                "tle": {
                    "line1": sat["TLE"]["line1"],
                    "line2": sat["TLE"]["line2"]
                }
            },
            "payloadType": payload_type,
            "beamCoverageKm": 2000,
            "islCapable": True,
            "bandwidth": 20000,
            "bandwidthMbps": 20000,
            "latencyMs": 30,
            "linkAvailable": True,
            "healthy": True,
            "lastUpdated": sat['EPOCH'],
            "velocity": velocity, # vx/vy/vz (km/s)
        }
        
        # validate theo schema
        validate(instance=item, schema=SCHEMA)
        return item
    except Exception as e:
        print(f"Error processing satellite {sat['OBJECT_ID'], sat['OBJECT_NAME']}: {e}")
        return None

def main():
    processed = []
    
    for sat in RAW:
        item = preprocess_satellite(sat)
        if item:
            processed.append(item)
        
    # Lưu kết quả
    OUT.parent.mkdir(parents=True, exist_ok=True) # tạo thư mục nếu chưa tồn tại
    OUT.write_text(json.dumps(processed, indent=2), encoding="utf-8")
    print("Đã xử lý và lưu dữ liệu vệ tinh ->", OUT)
    
if __name__ == "__main__":
    main()
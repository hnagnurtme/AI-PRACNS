from db_config import get_collection
nodes = get_collection("nodes")

from datetime import datetime
import math
import time
import random
import logging

# --- 1. Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- 2. Database setup ---

# --- 3. Hàm cập nhật vị trí ---
def update_positions(interval_seconds=5):
    logging.info("Node position updater started.")
    while True:
        try:
            all_nodes = list(nodes.find({}))
            for node in all_nodes:
                pos = node.get("position", {})
                vel = node.get("velocity", {})
                node_type = node.get("type", "")

                lat = pos.get("latitude", 0.0)
                lon = pos.get("longitude", 0.0)
                alt = pos.get("altitude", 0.0)

                vx = vel.get("velocityX", 0.0)
                vy = vel.get("velocityY", 0.0)
                vz = vel.get("velocityZ", 0.0)

                # --- Nếu là vệ tinh ---
                if "SATELLITE" in node_type:
                    lat += vx * 0.01 + math.sin(time.time() / 60) * 0.001
                    lon += vy * 0.01 + math.cos(time.time() / 60) * 0.001
                    alt += vz * 0.1 + random.uniform(-0.05, 0.05)
                else:
                    lat += random.uniform(-0.00001, 0.00001)
                    lon += random.uniform(-0.00001, 0.00001)

                # --- Giới hạn ---
                lat = max(-90, min(90, lat))
                lon = (lon + 180) % 360 - 180

                nodes.update_one(
                    {"_id": node["_id"]},
                    {"$set": {
                       # ipAddress:localhost
                        "communication.ipAddress": "127.0.0.1",
                        "position.latitude": lat,
                        "position.longitude": lon,
                        "position.altitude": alt,
                        "status.lastUpdated": datetime.utcnow().isoformat() + "Z"
                    }}
                )

            logging.info(f"Updated {len(all_nodes)} nodes.")
            time.sleep(interval_seconds)

        except Exception as e:
            logging.error(f"Error during update: {e}")
            time.sleep(2)  # tránh spam lỗi quá nhanh

# --- 4. Entry point ---
if __name__ == "__main__":
    update_positions()

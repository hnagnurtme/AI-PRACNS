import time
import random
from datetime import datetime
from pymongo import MongoClient

# --- Configuration ---
LOCAL_MONGO_URI = "mongodb://user:password123@localhost:27018/"
DB_NAME = "sagsin_network"
UPDATE_INTERVAL_SECONDS = 10 # Update satellite positions every 10 seconds

def get_mongo_client(uri):
    """Creates a MongoClient from a URI."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster') # Check connection
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB at {uri}: {e}")
        return None

def update_satellite_positions(client):
    """
    Updates the positions of LEO, MEO, and GEO satellites in the database.
    - LEO/MEO: Simulate movement by incrementing longitude.
    - GEO: Keep fixed (or very slight random wobble).
    """
    if not client:
        print("No MongoDB client available for update.")
        return

    db = client[DB_NAME]
    nodes_collection = db["network_nodes"]

    print(f"[{datetime.now()}] Updating satellite positions...")

    # Fetch all satellite nodes
    satellites = nodes_collection.find({
        "nodeType": {"$in": ["LEO_SATELLITE", "MEO_SATELLITE", "GEO_SATELLITE"]}
    })

    for sat in satellites:
        node_id = sat["nodeId"]
        node_type = sat["nodeType"]
        current_pos = sat["position"]

        new_longitude = current_pos["longitude"]
        new_latitude = current_pos["latitude"]
        new_altitude = current_pos["altitude"]

        if node_type == "LEO_SATELLITE":
            # LEOs move faster
            new_longitude = (new_longitude + random.uniform(0.5, 1.5)) % 360
            new_latitude = (new_latitude + random.uniform(-0.1, 0.1)) # Slight latitude drift
        elif node_type == "MEO_SATELLITE":
            # MEOs move moderately
            new_longitude = (new_longitude + random.uniform(0.1, 0.5)) % 360
            new_latitude = (new_latitude + random.uniform(-0.05, 0.05)) # Slight latitude drift
        elif node_type == "GEO_SATELLITE":
            # GEOs are geostationary, so minimal movement
            new_longitude = (new_longitude + random.uniform(-0.001, 0.001)) % 360
            new_latitude = (new_latitude + random.uniform(-0.001, 0.001)) # Very slight wobble

        # Ensure latitude stays within bounds
        new_latitude = max(-90, min(90, new_latitude))

        nodes_collection.update_one(
            {"nodeId": node_id},
            {"$set": {
                "position.longitude": new_longitude,
                "position.latitude": new_latitude,
                "lastUpdated": datetime.now() # Update timestamp
            }}
        )
        # print(f"  - Updated {node_id} ({node_type}): Lat={new_latitude:.2f}, Lon={new_longitude:.2f}")

    print(f"[{datetime.now()}] Satellite position update complete.")

if __name__ == "__main__":
    print("Starting satellite position update scheduler (local-only mode)...")
    mongo_client = get_mongo_client(LOCAL_MONGO_URI)

    if not mongo_client:
        print("Failed to connect to local MongoDB. Exiting scheduler.")
    else:
        try:
            while True:
                update_satellite_positions(mongo_client)
                time.sleep(UPDATE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
        finally:
            mongo_client.close()
            print("MongoDB connection closed.")
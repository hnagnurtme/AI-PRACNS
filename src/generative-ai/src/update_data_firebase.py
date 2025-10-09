import json
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path

# get project root (generative-ai) relative to this script
ROOT = Path(__file__).resolve().parents[1]  # go up 1 level from src/ to generative-ai/

# Initialize Firebase (check if already initialized)
try:
    app = firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(ROOT / "src" / "serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load updated ground station data
gs_file = ROOT / "data" / "processed" / "ground_station" / "ground_station_processed.json"

if not gs_file.exists():
    print(f"ERROR: {gs_file} not found")
    exit(1)

ground_stations = json.loads(gs_file.read_text(encoding="utf-8"))
print(f"Loaded {len(ground_stations)} ground stations from file")

# Update Firebase collection
updated_count = 0
failed_count = 0

for station in ground_stations:
    try:
        node_id = station["nodeId"]
        
        # Update only specific fields that might have changed
        update_data = {
            "healthy": station["healthy"],
        }
        
        # Update document in Firebase
        doc_ref = db.collection("ground_stations").document(node_id)
        doc_ref.update(update_data)
        
        updated_count += 1
        print(f"✅ Updated {node_id}: healthy={station['healthy']}")
        
    except Exception as e:
        failed_count += 1
        print(f"❌ Failed to update {station.get('nodeId', 'unknown')}: {e}")

print(f"\n📊 Summary:")
print(f"✅ Successfully updated: {updated_count}")
print(f"❌ Failed updates: {failed_count}")
print(f"📁 Total processed: {len(ground_stations)}")
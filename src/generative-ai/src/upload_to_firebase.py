import json
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path

# get project root (generative-ai) relative to this script
ROOT = Path(__file__).resolve().parents[1]  # go up 1 level from src/ to generative-ai/

cred = credentials.Certificate(ROOT / "src" / "serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

sat_file = ROOT / "data" / "processed" / "satellites" / "satellites_processed_01.json"
gs_file = ROOT / "data" / "processed" / "ground_station" / "ground_station_processed.json"
sea_file = ROOT / "data" / "processed" / "sea_station" / "sea_station_processed.json"

# check if files exist before reading
if not sat_file.exists():
    print(f"WARNING: {sat_file} not found, skipping satellites")
    satellites = []
else:
    satellites = json.loads(sat_file.read_text(encoding="utf-8"))

if not gs_file.exists():
    print(f"WARNING: {gs_file} not found, skipping ground stations")
    ground_stations = []
else:
    ground_stations = json.loads(gs_file.read_text(encoding="utf-8"))

if not sea_file.exists():
    print(f"WARNING: {sea_file} not found, skipping sea stations")
    sea_stations = []
else:
    sea_stations = json.loads(sea_file.read_text(encoding="utf-8"))

# upload to firebase
for sat in satellites:
    doc_id = sat["nodeId"]
    db.collection("satellites").document(doc_id).set(sat)

for gs in ground_stations:
    doc_id = gs["nodeId"]
    db.collection("ground_stations").document(doc_id).set(gs)

for sea in sea_stations:
    doc_id = sea["nodeId"]
    db.collection("sea_stations").document(doc_id).set(sea)

print("Data uploaded to Firebase Firestore successfully.")
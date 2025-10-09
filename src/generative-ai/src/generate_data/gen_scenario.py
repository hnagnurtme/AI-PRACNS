import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from uuid import uuid4
from jsonschema import validate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import requests
import math

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

# Đường dẫn output
ROOT = Path(__file__).resolve().parents[2]  # lên 2 cấp về generative-ai/
OUT_PATH = ROOT / "data" / "scenarios"
SCHEMA_SCENARIO = ROOT / "data" / "schema" / "scenario.json"

# Timestamp bắt đầu từ hiện tại (29/9/2025, 01:15 AM +07:00)
CURRENT_TIME = datetime.now(timezone.utc).replace(hour=18, minute=15, second=0, microsecond=0)  # 01:15 AM UTC
TIMESTAMP = int(CURRENT_TIME.timestamp() * 1000)

def load_llm():
    model_name = "gpt2-medium"  # Thay LLaMA-2 bằng GPT-2 (không cần token)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return pipeline("text-generation", model=model, tokenizer=tokenizer, 
                       pad_token_id=tokenizer.eos_token_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using fallback scenario generation without LLM")
        return None

gen = load_llm()

def generate_random_ue(timestamp: int, nodes: list) -> list:
    ues = []
    num_ues = random.randint(5, 10)
    mobility_patterns = ["static", "pedestrian", "vehicle", "aircraft"]
    traffic_types = ["video", "voice", "web", "iot"]
    
    for _ in range(num_ues):
        mobility = random.choice(mobility_patterns)
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        altitude = 0 if mobility in ["static", "pedestrian"] else random.uniform(0, 10) if mobility == "vehicle" else random.uniform(0.1, 10)
        velocity = {
            "vx": 0 if mobility == "static" else random.uniform(-10, 10) if mobility == "pedestrian" else random.uniform(-50, 50) if mobility == "vehicle" else random.uniform(-200, 200),
            "vy": 0 if mobility == "static" else random.uniform(-10, 10) if mobility == "pedestrian" else random.uniform(-50, 50) if mobility == "vehicle" else random.uniform(-200, 200),
            "vz": 0 if mobility == "static" else random.uniform(-1, 1) if mobility == "pedestrian" else random.uniform(-5, 5) if mobility == "vehicle" else random.uniform(-50, 50)
        }
        
        weather_response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=precipitation,cloudcover"
        ).json()
        # Tính attenuation_dB và snrReduction_dB dựa trên precipitation_mm_h
        
        weather = {
            "precipitation_mm_h": weather_response["hourly"]["precipitation"][0],
            "cloud_cover_percent": weather_response["hourly"]["cloudcover"][0],
            "attenuation_dB": 0.0,
            "snrReduction_dB": 0.0,
            "snr_dB": 25.0
        }
        
        pathLengthKm = float("inf")
        for node in nodes:
            dist = math.sqrt(
                (latitude - node["position"]["latitude"]) ** 2 +
                (longitude - node["position"]["longitude"]) ** 2
            ) * 111
            pathLengthKm = min(pathLengthKm, dist)
        latencyMs = pathLengthKm / 0.3
        
        ue = {
            "nodeId": f"ue_{uuid4()}",
            "nodeType": "UE",
            "position": {"latitude": latitude, "longitude": longitude, "altitude": altitude},
            "velocity": velocity,
            "bandwidth": random.uniform(1, 50),
            "healthy": random.random() < 0.9,
            "linkAvailable": random.random() < 0.9,
            "lastUpdated": datetime.utcfromtimestamp(timestamp / 1000).isoformat() + "+00:00",
            "weather": weather,
            "pathLengthKm": pathLengthKm,
            "latencyMs": latencyMs,
            "service": {
                "trafficType": random.choice(traffic_types),
                "requiredBandwidthMbps": random.uniform(1, 20) if random.choice(traffic_types) in ["video", "voice"] else random.uniform(0.1, 5),
                "maxLatencyMs": random.randint(50, 500) if random.choice(traffic_types) in ["video", "voice"] else random.randint(100, 1000),
                "priorityLevel": random.randint(1, 5)
            },
            "energyLevel": random.uniform(20, 100),
            "sessionDurationSec": random.randint(300, 3600),
            "mobilityPattern": mobility
        }
        ues.append(ue)
    return ues

def get_nodes(timestamp: int) -> list:
    satellites = [doc.to_dict() for doc in db.collection("satellites").where("lastUpdated", "<=", timestamp).stream()]
    ground_stations = [doc.to_dict() for doc in db.collection("ground_stations").where("lastUpdated", "<=", timestamp).stream()]
    sea_stations = [doc.to_dict() for doc in db.collection("sea_stations").where("lastUpdated", "<=", timestamp).stream()]
    return satellites + ground_stations + sea_stations

def generate_scenario_with_llm(timestamp: int, previous_scenario: dict = None) -> dict:
    nodes = get_nodes(timestamp) + generate_random_ue(timestamp, get_nodes(timestamp))
    nodes_json = json.dumps(nodes, indent=2)
    
    if previous_scenario:
        previous_scenario_json = json.dumps(previous_scenario, indent=2)
        prompt = f"""
Bạn là GenAI chuyên sinh kịch bản cho mạng SAGINs. Dữ liệu đầu vào là kịch bản trước và timestamp mới.

Nhiệm vụ: Sinh kịch bản mới theo schema scenario.json, dựa trên kịch bản trước, với timestamp mới.

Cập nhật nodes: Gồm satellite, ground station, sea station, và UE. Thay đổi healthy/linkAvailable ngẫu nhiên (5-15% node thành healthy=false), giữ nguyên vị trí, thời tiết, pathLengthKm, latencyMs từ dữ liệu đầu vào.

Đa dạng hóa: Sinh events mới (node_failure, traffic_spike, weather_change).

Trả về chỉ JSON kịch bản mới, không giải thích.

Kịch bản trước: {previous_scenario_json}

Timestamp mới: {timestamp}
"""
    else:
        prompt = f"""
Bạn là GenAI chuyên sinh kịch bản cho mạng SAGINs. Dữ liệu đầu vào là danh sách nodes (satellite, ground station, sea station, UE) với thông tin vị trí, bandwidth, latencyMs, weather, v.v.

Nhiệm vụ: Sinh một kịch bản theo schema scenario.json:
{{
  "scenarioId": "sagin_<UUID>",
  "timestamp": {timestamp},
  "nodes": [mảng các node đầu vào, thay đổi ngẫu nhiên 10-20% node thành healthy=false],
  "trafficProfile": {{
    "flows": <số luồng, 10-50 * số node>,
    "mix": {{"video": 0.3, "voice": 0.3, "web": 0.2, "iot": 0.2}}  # Thay đổi ngẫu nhiên ±0.1
  }},
  "events": [mảng 2-5 sự kiện: type (node_failure, traffic_spike, weather_change), target (nodeId), start (timestamp + 300000-7200000 ms), durationSec (300-1800 giây)],
  "routes": []
}}

Dữ liệu nodes đầu vào: {nodes_json}

Đa dạng hóa: Thay đổi healthy/linkAvailable ngẫu nhiên, sinh events khác nhau mỗi lần.

Trả về chỉ JSON kịch bản, không giải thích.
"""
    
    response = gen(prompt, max_length=2000, num_return_sequences=1)[0]["generated_text"]
    
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        scenario_json = response[json_start:json_end]
        scenario = json.loads(scenario_json)
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}")
        scenario = generate_scenario_fallback(timestamp, nodes)
    
    validate(instance=scenario, schema=json.loads(SCHEMA_SCENARIO.read_text(encoding="utf-8")))
    return scenario

def generate_scenario_fallback(timestamp: int, nodes: list) -> dict:
    for node in nodes:
        node["healthy"] = random.random() < 0.85
        node["linkAvailable"] = node["healthy"] and (random.random() < 0.85)
        node["bandwidth"] = node["bandwidth"] * random.uniform(0.8, 1.2)
    
    total_nodes = len(nodes)
    flows = random.randint(10, 50) * total_nodes
    mix_weights = [random.uniform(0.2, 0.4), random.uniform(0.2, 0.4), random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)]
    mix_sum = sum(mix_weights)
    mix = {
        "video": round(mix_weights[0] / mix_sum, 2),
        "voice": round(mix_weights[1] / mix_sum, 2),
        "web": round(mix_weights[2] / mix_sum, 2),
        "iot": round(mix_weights[3] / mix_sum, 2)
    }
    
    events = []
    failure_nodes = random.sample([n["nodeId"] for n in nodes], k=max(1, int(0.1 * total_nodes)))
    for node_id in failure_nodes:
        events.append({
            "type": "node_failure",
            "target": node_id,
            "start": timestamp + random.randint(300000, 3600000),
            "durationSec": random.randint(300, 1800)
        })
    spike_nodes = random.sample([n["nodeId"] for n in nodes], k=min(3, total_nodes))
    for node_id in spike_nodes:
        events.append({
            "type": "traffic_spike",
            "target": node_id,
            "start": timestamp + random.randint(300000, 7200000),
            "durationSec": random.randint(300, 3600)
        })
    weather_nodes = random.sample([n["nodeId"] for n in nodes], k=max(1, int(0.1 * total_nodes)))
    for node_id in weather_nodes:
        events.append({
            "type": "weather_change",
            "target": node_id,
            "start": timestamp + random.randint(300000, 7200000),
            "durationSec": random.randint(600, 3600)
        })
    
    scenario = {
        "scenarioId": f"sagin_{uuid4()}",
        "timestamp": timestamp,
        "nodes": nodes,
        "trafficProfile": {"flows": flows, "mix": mix},
        "events": events,
        "routes": []
    }
    
    return scenario

def on_snapshot_callback(col_snapshot, changes, read_time):
    timestamp = int(read_time.timestamp() * 1000)
    scenario = generate_scenario_with_llm(timestamp)
    out_file = OUT_PATH / f"scenario_{timestamp}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"Đã lưu kịch bản vào {out_file}")
    
    doc_ref = db.collection("scenarios").document(scenario["scenarioId"])
    doc_ref.set(scenario)
    print(f"Đã lưu kịch bản vào Firestore: scenarios/{scenario['scenarioId']}")

def main():
    collections = ["satellites", "ground_stations", "sea_stations"]
    for collection in collections:
        db.collection(collection).on_snapshot(on_snapshot_callback)
    
    timestamps = [TIMESTAMP + i * 300000 for i in range(10)]
    previous_scenario = None
    for ts in timestamps:
        scenario = generate_scenario_with_llm(ts, previous_scenario)
        out_file = OUT_PATH / f"scenario_{ts}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        print(f"Đã lưu kịch bản vào {out_file}")
        
        doc_ref = db.collection("scenarios").document(scenario["scenarioId"])
        doc_ref.set(scenario)
        print(f"Đã lưu kịch bản vào Firestore: scenarios/{scenario['scenarioId']}")
        previous_scenario = scenario

if __name__ == "__main__":
    main()
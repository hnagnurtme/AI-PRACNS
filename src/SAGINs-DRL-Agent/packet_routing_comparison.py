# check_gs_distance.py
from pymongo import MongoClient
import math

mongo = MongoClient('mongodb://user:password123@localhost:27017/?authSource=admin')
db = mongo['sagsin_network']

def haversine_distance(pos1, pos2):
    """Calculate distance in km"""
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(pos1['latitude']), math.radians(pos1['longitude'])
    lat2, lon2 = math.radians(pos2['latitude']), math.radians(pos2['longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

gs1 = db.nodes.find_one({"nodeId": "GS-01"})
gs2 = db.nodes.find_one({"nodeId": "GS-02"})

if gs1 and gs2:
    dist = haversine_distance(gs1['position'], gs2['position'])
    print(f"📏 Distance GS-01 ↔ GS-02: {dist:.2f} km")
    print(f"📍 GS-01: lat={gs1['position']['latitude']:.4f}, lon={gs1['position']['longitude']:.4f}")
    print(f"📍 GS-02: lat={gs2['position']['latitude']:.4f}, lon={gs2['position']['longitude']:.4f}")
    print(f"\n🔧 Current GS maxRange: {gs1['communication']['maxRangeKm']} km")
    
    if dist < 800:
        print(f"❌ Distance ({dist:.2f} km) < maxRange (800 km) → Direct connection possible!")
        print(f"💡 Solution: Set maxRangeKm to {int(dist * 0.8)} km or less")
    else:
        print(f"✅ Distance ({dist:.2f} km) > maxRange (800 km) → Must use satellites")
else:
    print("❌ Ground Stations not found")

mongo.close()
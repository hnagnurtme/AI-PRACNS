import requests, json, random
from pathlib import Path

URL_JSON = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
URL_TLE  = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
OUT = Path("../../data/raw/satellites/satellites_raw_01.json")

# --- Lấy JSON (thông số GP)
data = requests.get(URL_JSON, timeout=15).json()

# --- Lấy text TLE (3 dòng cho mỗi vệ tinh)
tle_text = requests.get(URL_TLE, timeout=15).text.strip().splitlines()

# Gom theo nhóm 3 dòng: [name, line1, line2]
tle_groups = [tle_text[i:i+3] for i in range(0, len(tle_text), 3)]
tle_map = {g[0].strip(): {"line1": g[1].strip(), "line2": g[2].strip()} for g in tle_groups}

# --- Chọn ngẫu nhiên 60 vệ tinh
subset = random.sample(data, 60)

# --- Ghép TLE vào từng object (nếu trùng tên)
for sat in subset:
    name = sat.get("OBJECT_NAME","").strip()
    if name in tle_map:
        sat["TLE"] = tle_map[name]

OUT.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
print("Đã tải dữ liệu vệ tinh + TLE ->", OUT)

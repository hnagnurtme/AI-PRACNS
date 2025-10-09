import sys
import requests
import json
import re
import html
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math, os

URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "data" / "raw" / "sea_stations" / "sea_stations_raw_01.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# simple session with retries
session = requests.Session()
session.trust_env = True
retries = Retry(total=3, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

try:
    res = session.get(URL, timeout=(5, 30))
    res.raise_for_status()
    text = res.text
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
    sys.exit(1)

lines = [ln for ln in text.splitlines() if ln.strip()]

# debug: show what we fetched
print(f"DEBUG: fetched {len(lines)} lines from {URL}")
for i, ln in enumerate(lines[:8]):
    print(f"DEBUG line {i+1}: {ln[:200]}")

# header detection from commented header line, example:
# STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD | LOCATION | TIMEZONE | FORECAST | NOTE
header_map = {}
for ln in lines:
    if ln.lstrip().startswith("#"):
        candidate = ln.lstrip("#").strip()
        if "|" in candidate:
            parts = [p.strip().lower() for p in candidate.split("|")]
            for i, p in enumerate(parts):
                # normalize keys
                key = re.sub(r"[^\w]", "_", p).strip("_")
                header_map[key] = i
            break

def split_row(line):
    # prefer pipe-separated; fallback to columns separated by 2+ spaces or tabs
    if "|" in line:
        return [p.strip() for p in line.split("|")]
    parts = [p.strip() for p in re.split(r"\s{2,}|\t", line) if p.strip()]
    return parts

def clean_html(s):
    if s is None:
        return None
    s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip() or None

latlon_re = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*([NS])\s+([+-]?\d+(?:\.\d+)?)\s*([EW])", re.I)
alt_decimal_re = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*[°,]?\s*([NS])[,;\s]+\s*([+-]?\d+(?:\.\d+)?)\s*[°,]?\s*([EW])", re.I)
decimal_pair_re = re.compile(r"([+-]?\d+(?:\.\d+))\s*[°,]?\s*[NSns]?[,;\s()/-]+\s*([+-]?\d+(?:\.\d+))\s*[°,]?\s*[EWew]?", re.I)
any_two_floats = re.compile(r"([+-]?\d+(?:\.\d+)).{0,12}?([+-]?\d+(?:\.\d+))")

def parse_latlon(loc):
    if not loc:
        return None, None
    # 1) common patterns with N/S and E/W
    m = latlon_re.search(loc)
    if not m:
        m = alt_decimal_re.search(loc)
    if not m:
        m = decimal_pair_re.search(loc)
    if not m:
        # try to find two floats close to each other (fallback)
        m2 = any_two_floats.search(loc)
        if m2:
            try:
                a = float(m2.group(1)); b = float(m2.group(2))
                # Heuristic: lat in [-90,90], lon in [-180,180]
                if -90 <= a <= 90 and -180 <= b <= 180:
                    return a, b
                if -90 <= b <= 90 and -180 <= a <= 180:
                    return b, a
            except Exception:
                pass
        return None, None

    lat_val = float(m.group(1))
    ns = m.group(2).upper() if m.group(2) else None
    lon_val = float(m.group(3))
    ew = m.group(4).upper() if m.group(4) else None

    # if N/S/E/W present, use them
    if ns and ew:
        lat = -lat_val if ns == "S" else lat_val
        lon = -lon_val if ew == "W" else lon_val
        return lat, lon

    # last resort: assume first is lat, second is lon
    return lat_val, lon_val

# fallback default order if header not present
defaults = {
    "station_id": 0,
    "owner": 1,
    "ttype": 2,
    "hull": 3,
    "name": 4,
    "payload": 5,
    "location": 6,
    "timezone": 7,
    "forecast": 8,
    "note": 9,
}

def get_field(parts, keys, default_index):
    for k in keys:
        if k in header_map and header_map[k] < len(parts):
            return parts[header_map[k]] or None
    if default_index is not None and default_index < len(parts):
        return parts[default_index] or None
    return None

records = []
for ln in lines:
    if ln.lstrip().startswith("#"):
        continue
    parts = split_row(ln)
    if len(parts) == 0:
        continue

    station_id = get_field(parts, ["station_id", "station_id_", "station"], defaults["station_id"])
    owner = get_field(parts, ["owner"], defaults["owner"])
    ttype = get_field(parts, ["ttype", "ttype_"], defaults["ttype"])
    hull = get_field(parts, ["hull"], defaults["hull"])
    name = get_field(parts, ["name"], defaults["name"])
    payload = get_field(parts, ["payload"], defaults["payload"])
    location = get_field(parts, ["location"], defaults["location"])
    timezone = get_field(parts, ["timezone"], defaults["timezone"])
    forecast = get_field(parts, ["forecast"], defaults["forecast"])
    note = get_field(parts, ["note"], defaults["note"])

    lat, lon = parse_latlon(location or "")
    rec = {
        "station_id": station_id,
        "owner": owner,
        "type": ttype,
        "hull": hull,
        "name": clean_html(name),
        "payload": payload,
        "location_raw": clean_html(location),
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast": forecast,
        "note": clean_html(note),
    }
    records.append(rec)

if not records:
    print("Không tìm thấy bản ghi hợp lệ để lưu.")
    sys.exit(1)

# ---------- sampling: chọn số trạm giới hạn, phân bố đều toàn cầu ----------
# fixed sample count (set here)
SAMPLE_COUNT = 120

if SAMPLE_COUNT and SAMPLE_COUNT > 0:
    # chỉ dùng các bản ghi có lat/lon hợp lệ
    valid = [r for r in records if r.get("latitude") is not None and r.get("longitude") is not None]
    if valid:
        rows = max(1, int(math.sqrt(SAMPLE_COUNT)))
        cols = math.ceil(SAMPLE_COUNT / rows)
        # tạo lưới lat từ -80..80 (tránh điểm cực), lon từ -180..180
        lats = [ -80 + i * (160 / (rows - 1)) if rows > 1 else 0 for i in range(rows) ]
        lons = [ -180 + j * (360 / (cols - 1)) if cols > 1 else 0 for j in range(cols) ]

        targets = []
        for lat in lats:
            for lon in lons:
                targets.append((lat, lon))
                if len(targets) >= SAMPLE_COUNT:
                    break
            if len(targets) >= SAMPLE_COUNT:
                break

        selected = []
        seen = set()
        # ensure haversine exists; if not, fallback to simple euclidean on lat/lon
        def _dist(a_lat, a_lon, b_lat, b_lon):
            try:
                return haversine(a_lat, a_lon, b_lat, b_lon)
            except NameError:
                return math.hypot(a_lat - b_lat, a_lon - b_lon)

        for tlat, tlon in targets:
            # tìm trạm gần nhất tới điểm lưới
            best = min(valid, key=lambda r: _dist(tlat, tlon, r["latitude"], r["longitude"]))
            sid = best.get("station_id") or best.get("name") or f"{best.get('latitude')},{best.get('longitude')}"
            if sid in seen:
                continue
            seen.add(sid)
            selected.append(best)
            if len(selected) >= SAMPLE_COUNT:
                break

        records = selected

# after building records list, add quick debug
print(f"DEBUG: parsed {len(records)} candidate records (before sampling).")
if len(records) <= 10:
    for r in records[:20]:
        print("DEBUG record:", r)
else:
    print("DEBUG sample record 1:", records[0])

if not OUT.parent.exists():
    print(f"❌ Thư mục lưu trữ không tồn tại: {OUT.parent}")
    sys.exit(1)

# ensure file is written at end
OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"✅ Đã parse {len(records)} trạm → {OUT}")

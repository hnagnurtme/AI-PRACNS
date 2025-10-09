import sys
import json
import random
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://network.satnogs.org/api/stations/"

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "data" / "raw" / "satellites" / "ground_station_raw.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# session with retries and backoff
session = requests.Session()
session.trust_env = True  # allow system proxy env vars if present
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET", "HEAD"),
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

try:
    # timeout = (connect_timeout, read_timeout)
    res = session.get(URL, timeout=(5, 60))
    res.raise_for_status()
    data = res.json()
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
    sys.exit(1)
except json.JSONDecodeError as e:
    print("Invalid JSON received:", e)
    sys.exit(1)

# sample safely: nếu data là list và có đủ phần tử, lấy ngẫu nhiên n, ngược lại lưu nguyên
n = 100
if isinstance(data, list):
    if len(data) >= n:
        subset = random.sample(data, n)
    else:
        subset = data
else:
    # nếu server trả dict hoặc khác, lưu dưới dạng list để dễ dùng sau
    subset = [data]

OUT.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
print("Đã tải và lưu:", OUT)
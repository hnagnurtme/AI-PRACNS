import numpy as np

# --- Vật lý cơ bản ---
SPEED_OF_LIGHT = 299792.458  # km/s
EARTH_RADIUS_KM = 6371.0     # Bán kính trái đất trung bình
BOLTZMANN_CONST = -228.6     # dBW/K/Hz
NOISE_TEMP_K = 290.0         # Kelvin (Standard noise temperature)

# --- Packet & Data ---
DEFAULT_PACKET_SIZE_BYTES = 1500  # Kích thước gói tin giả định (MTU)
BITS_IN_BYTE = 8

# --- Normalization Bounds (Scale về [0,1]) ---
MAX_DIST_KM = 40000.0        # GEO orbit (giữ nguyên nếu có GEO, giảm nếu chỉ LEO)
MAX_QUEUE_CAPACITY = 10000   # Max buffer size
DEFAULT_BUFFER_CAPACITY = 4000 # Tăng lên cho sát thực tế mẫu (4040)

MAX_BW_MHZ = 1000.0          # Max Bandwidth
MIN_SNR_DB = -10.0           # Ngưỡng mất tín hiệu
MAX_SNR_DB = 40.0            # Tín hiệu cực tốt

# --- Delays ---
MAX_PROCESSING_DELAY_MS = 1000.0 # Delay xử lý nội tại (Queue + CPU)
MAX_PROPAGATION_DELAY_MS = 500.0 # Delay truyền dẫn (Distance / C) - Đổi tên cho rõ

# --- RL Configuration ---
MAX_NEIGHBORS = 10
NEIGHBOR_FEAT_SIZE = 14


WEATHER_LOSS_MAP = {
    "CLEAR": 0.0,
    "CLOUDY": 1.0,
    "RAIN": 8.0,     # Trung bình mưa
    "HEAVY_RAIN": 15.0,
    "STORM": 25.0
}
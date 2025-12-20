"""
Constants for Routing Environment and State Builder
Centralized configuration to avoid magic numbers
"""
import math

# ============================================================================
# DISTANCE CONSTANTS
# ============================================================================
EARTH_RADIUS_M = 6371000
M_TO_KM = 1000.0
KM_TO_M = 1000.0

# Distance thresholds (in meters)
DISTANCE_NEAR_DEST_M = 500000  # 500km
DISTANCE_CLOSE_DEST_M = 1000000  # 1000km
DISTANCE_FAR_DEST_M = 2000000  # 2000km
DISTANCE_VERY_CLOSE_M = 1000  # 1km

# Distance normalization factors
DIST_NORM_MAX_M = 20000000  # 20,000km
DIST_NORM_CURRENT_M = 1000000  # 1000km
DIST_NORM_EDGE_WEIGHT_KM = 10000.0

# ============================================================================
# SPEED OF LIGHT
# ============================================================================
SPEED_OF_LIGHT_MPS = 299792458
MS_PER_SECOND = 1000.0

# ============================================================================
# RESOURCE UTILIZATION CONSTANTS
# ============================================================================
UTILIZATION_MAX_PERCENT = 100.0
UTILIZATION_CRITICAL_PERCENT = 95.0
UTILIZATION_HIGH_PERCENT = 80.0
UTILIZATION_MEDIUM_PERCENT = 70.0
UTILIZATION_LOW_PERCENT = 60.0

# Terminal connection impact
TERMINAL_UTILIZATION_IMPACT = 7.0  # Each terminal adds ~7% utilization

# ============================================================================
# NODE QUALITY CONSTANTS
# ============================================================================
BATTERY_LOW_PERCENT = 20.0
BATTERY_MAX_PERCENT = 100.0
PACKET_LOSS_HIGH = 0.1

# Ground station connection thresholds
GS_CONNECTION_OVERLOADED = 15
GS_CONNECTION_HIGH = 10
GS_CONNECTION_LOW = 3

# ============================================================================
# SCORING WEIGHTS
# ============================================================================
SCORE_WEIGHT_DIST_TO_DEST = 0.7
SCORE_WEIGHT_DIST_TO_CURRENT = 0.1
SCORE_WEIGHT_VISITED_PENALTY = 1000000.0
SCORE_WEIGHT_QUALITY = 500000.0

# Node type bonuses/penalties
BONUS_LEO_SATELLITE = -100000.0
BONUS_SATELLITE = -80000.0
BONUS_GS_LOAD_BALANCING = -100000.0
PENALTY_GS_TO_GS = 80000.0
PENALTY_GS_OVERLOADED = 200000.0
PENALTY_GS_HIGH_LOAD = 100000.0
BONUS_GS_LOW_LOAD = -80000.0

# Problem penalties
PENALTY_CRITICAL_UTILIZATION = 400000.0
PENALTY_HIGH_UTILIZATION = 250000.0
PENALTY_MEDIUM_UTILIZATION = 120000.0
PENALTY_LOW_BATTERY = 200000.0
PENALTY_HIGH_PACKET_LOSS = 100000.0

# ============================================================================
# REWARD CONSTANTS
# ============================================================================
REWARD_SUCCESS = 200.0
REWARD_FAILURE = -10.0
REWARD_STEP_PENALTY = -10.0
REWARD_HOP_PENALTY = -15.0
REWARD_GS_HOP_PENALTY = -25.0
REWARD_LOAD_BALANCING = 5.0
REWARD_LOOP_PENALTY = -20.0
REWARD_DROP_NODE = -1000.0

# Progress and distance scales
PROGRESS_REWARD_SCALE = 80.0
DISTANCE_REWARD_SCALE = 10.0
QUALITY_REWARD_SCALE = 10.0
PROXIMITY_BONUS_SCALE = 50.0

# Progress calculation
PROGRESS_DIVISOR_M = 100000.0
DETOUR_PENALTY_DIVISOR_M = 50000.0
DETOUR_PENALTY_MULTIPLIER = 30.0
DISTANCE_PENALTY_DIVISOR_M = 10000000.0

# Proximity thresholds
PROXIMITY_CLOSE_M = 1000000  # 1km
PROXIMITY_FAR_M = 2000000  # 2km
PROXIMITY_BONUS_MULTIPLIER = 2.0

# Success bonuses
BONUS_EXACT_DEST_GS = 50.0
BONUS_QOS_COMPLIANCE = 30.0
PENALTY_QOS_VIOLATION = -15.0

# Efficiency rewards
EFFICIENCY_BONUS_PER_HOP = 20.0
EFFICIENCY_PENALTY_PER_HOP = 15.0
EFFICIENCY_EXTRA_PENALTY_BASE = 5
EFFICIENCY_EXTRA_PENALTY_MULTIPLIER = 30.0

# Distance efficiency
DISTANCE_RATIO_EFFICIENT = 1.2
DISTANCE_RATIO_ACCEPTABLE = 1.5
DISTANCE_RATIO_POOR = 3.0
BONUS_DISTANCE_EFFICIENT = 30.0
BONUS_DISTANCE_ACCEPTABLE = 15.0
PENALTY_DISTANCE_POOR = -20.0

# Node quality thresholds
QUALITY_EXCELLENT = 0.8
QUALITY_GOOD = 0.6
QUALITY_BAD = 0.3
BONUS_EXCELLENT_NODE = 5.0
BONUS_GOOD_NODE = 3.0
PENALTY_BAD_NODE = -20.0

# Excess hops
EXCESS_HOPS_THRESHOLD = 3
EXCESS_HOPS_PENALTY_MULTIPLIER = 20.0

# ============================================================================
# FEATURE NORMALIZATION CONSTANTS
# ============================================================================
NORM_UTILIZATION = 100.0
NORM_PACKET_BUFFER = 1000
NORM_PROCESSING_DELAY_MS = 50.0
NORM_BANDWIDTH_MBPS = 1000.0
NORM_ALTITUDE_M = 50000.0
NORM_LATENCY_MS = 1000.0
NORM_BANDWIDTH_REQ_MBPS = 100.0
NORM_LOSS_RATE = 0.1
NORM_EDGE_WEIGHT_KM = 10.0

# ============================================================================
# NODE TYPE ENCODING
# ============================================================================
NODE_TYPE_SATELLITE = 0.2
NODE_TYPE_AERIAL = 0.5
NODE_TYPE_GROUND_STATION = 0.8

# ============================================================================
# QUALITY SCORE WEIGHTS
# ============================================================================
QUALITY_WEIGHT_RESOURCE = 0.3
QUALITY_WEIGHT_RELIABILITY = 0.3
QUALITY_WEIGHT_ENERGY = 0.2
QUALITY_WEIGHT_PERFORMANCE = 0.2

# ============================================================================
# COMMUNICATION RANGES
# ============================================================================
DEFAULT_MAX_RANGE_KM = 2000
SATELLITE_RANGE_MARGIN = 1.1  # 10% margin for orbital movement
GS_RANGE_MARGIN = 1.0  # No margin for ground stations

# ============================================================================
# PATH CONSTANTS
# ============================================================================
MIN_PATH_HOPS = 3
MIN_PATH_SEGMENTS = 4

# ============================================================================
# DIJKSTRA ALIGNMENT
# ============================================================================
DIJKSTRA_DROP_THRESHOLD = 95.0
DIJKSTRA_PENALTY_THRESHOLD = 80.0
DIJKSTRA_PENALTY_MULTIPLIER = 3.0
DIJKSTRA_PROGRESS_SCALE = 10.0

# ============================================================================
# PHASE 2: DYNAMIC MAX STEPS & PROGRESS DETECTION
# ============================================================================
PROGRESS_CHECK_MIN_STEPS = 3
PROGRESS_CHECK_WINDOW_SIZE = 3
PROGRESS_MIN_THRESHOLD_M = 1000.0
PROGRESS_NO_PROGRESS_PENALTY = -50.0
ADAPTIVE_MAX_STEPS_NETWORK_DIVISOR = 2
ADAPTIVE_MAX_STEPS_MULTIPLIER = 2

# ============================================================================
# PHASE 2: IMITATION LEARNING STRATIFIED SAMPLING
# ============================================================================
STRATIFIED_NEAR_RATIO = 0.3
STRATIFIED_MEDIUM_RATIO = 0.3
STRATIFIED_FAR_RATIO = 0.2
STRATIFIED_VERY_FAR_RATIO = 0.2
STRATIFIED_NEAR_DISTANCE_KM = 2000
STRATIFIED_MEDIUM_DISTANCE_KM = 5000
STRATIFIED_FAR_DISTANCE_KM = 10000
DEMO_SIMILARITY_THRESHOLD = 0.7
DEMO_LOG_FREQUENCY = 50

# Path quality thresholds
PATH_QUALITY_LONG_THRESHOLD = 10
PATH_QUALITY_MEDIUM_THRESHOLD = 7
PATH_QUALITY_SHORT_THRESHOLD = 5
PATH_QUALITY_LONG_MULTIPLIER = 0.5
PATH_QUALITY_MEDIUM_MULTIPLIER = 0.7
PATH_QUALITY_SHORT_MULTIPLIER = 0.9
PATH_QUALITY_UTIL_HIGH_THRESHOLD = 90
PATH_QUALITY_UTIL_MEDIUM_THRESHOLD = 80
PATH_QUALITY_UTIL_LOW_THRESHOLD = 60


import numpy as np

class MetricsCalculator:
    @staticmethod
    def calculate_routing_metrics(paths: list, delays: list, successes: list) -> dict:
        return {
            'success_rate': np.mean(successes),
            'avg_delay': np.mean(delays),
            'avg_hops': np.mean([len(path) for path in paths])
        }
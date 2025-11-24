class ComparisonUtils:
    @staticmethod
    def compare_algorithms(results_rl: dict, results_dijkstra: dict, results_baseline: dict) -> dict:
        comparison = {
            'RL': results_rl,
            'Dijkstra': results_dijkstra,
            'Baseline': results_baseline
        }
        return comparison
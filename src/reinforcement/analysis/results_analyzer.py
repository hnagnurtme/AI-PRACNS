class ResultsAnalyzer:
    def __init__(self, results: dict):
        self.results = results
        
    def generate_summary(self) -> dict:
        summary = {}
        for algo, metrics in self.results.items():
            summary[algo] = {
                'success_rate': metrics['success_rate'],
                'avg_delay': metrics['avg_delay'],
                'avg_hops': metrics['avg_hops']
            }
        return summary
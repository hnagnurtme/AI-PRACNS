class ReportGenerator:
    def __init__(self, results: dict):
        self.results = results
        
    def generate_report(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write("SAGIN Routing Algorithm Comparison Report\n")
            f.write("=========================================\n\n")
            for algo, metrics in self.results.items():
                f.write(f"{algo}:\n")
                f.write(f"  Success Rate: {metrics['success_rate']:.2f}\n")
                f.write(f"  Average Delay: {metrics['avg_delay']:.1f} ms\n")
                f.write(f"  Average Hops: {metrics['avg_hops']:.1f}\n\n")
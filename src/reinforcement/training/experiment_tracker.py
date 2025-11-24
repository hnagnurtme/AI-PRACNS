import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.metrics = []
        
    def log_metrics(self, episode: int, reward: float, loss: float, epsilon: float):
        self.metrics.append({
            'episode': episode,
            'reward': reward,
            'loss': loss,
            'epsilon': epsilon,
            'timestamp': datetime.now().isoformat()
        })
    
    def save_results(self, filepath: str):
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
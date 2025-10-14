# data/data_pipeline.py
import time 
import threading
from typing import Dict, Any
from .mongo_manager import MongoDataManager

class RealTimeDataPipeline:
    def __init__(self, mongo_manager: MongoDataManager, update_interval: int = 60):
        self.mongo_manager = mongo_manager
        self.update_interval = update_interval
        self._current_snapshot = None
        self._running = False
        self._lock = threading.Lock()
        
    def start(self):
        """Khởi động luồng cập nhật dữ liệu."""
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        print("Real-time data pipelint started.")
    
    def stop(self):
        """Dừng luồng cập nhật dữ liệu."""
        self._running = False
        print("Real-time data pipeline stopped.")
    
    def _update_loop(self):
        """Vòng lặp cập nhật dữ liệu"""  
        while self._running:
            try:
                new_snapshot = self.mongo_manager.get_training_snapshot()
                
                with self._lock:
                    self._current_snapshot = new_snapshot
                
                print(f"Data snapshot updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"Error fetching data from MongoDB: {e}")
            
            time.sleep(self.update_interval)
    
    def get_current_snapshot(self) -> Dict[str, Any]:
        """Lấy snapshot hiện tại"""
        with self._lock:
            if self._current_snapshot is None:
                return self.mongo_manager.get_training_snapshot()
            return self._current_snapshot
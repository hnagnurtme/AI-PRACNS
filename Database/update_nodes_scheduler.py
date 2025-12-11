#!/usr/bin/env python3
"""
Scheduled Node Position Update Service
Cập nhật vị trí satellites trong database theo lịch trình

Usage:
    python update_nodes_scheduler.py [--interval SECONDS] [--daemon]
    
Options:
    --interval SECONDS    Update interval in seconds (default: 10)
    --daemon             Run as daemon process
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime
from typing import List, Dict

# Add parent directory to path to import Backend modules
backend_path = os.path.join(os.path.dirname(__file__), '..', 'Backend')
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)

# Try to import from Backend, fallback to local if in Docker
try:
    from models.database import db
except ImportError:
    # If running in Docker or standalone, create minimal db connection
    import os
    from pymongo import MongoClient
    
    class SimpleDB:
        def __init__(self):
            self.client = None
            self.db = None
        
        def connect(self):
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://admin:password@localhost:27017/aiprancs?authSource=admin')
            db_name = os.getenv('DB_NAME', 'aiprancs')
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            return True
        
        def is_connected(self):
            try:
                if self.client:
                    self.client.admin.command('ping')
                    return True
                return False
            except:
                return False
        
        def get_collection(self, name):
            if self.db is None:
                self.connect()
            return self.db[name] if self.db is not None else None
        
        def close(self):
            if self.client:
                self.client.close()
    
    db = SimpleDB()
    db.connect()

from orbital_mechanics import OrbitalMechanics
from config_loader import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('node_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NodePositionScheduler:
    """Scheduled service để cập nhật vị trí nodes"""
    
    def __init__(self, update_interval: int = None, config_file: str = None):
        """
        Initialize scheduler
        
        Args:
            update_interval: Update interval in seconds (if None, read from config)
            config_file: Optional path to config file
        """
        self.config = get_config(config_file)
        
        # Get update interval from config or parameter
        self.update_interval = update_interval or self.config.get(
            'scheduler.update_interval_seconds', 10
        )
        
        # Initialize orbital engine with config
        self.orbital_engine = OrbitalMechanics(config_file)
        
        # Get node types to update from config
        self.update_node_types = self.config.get(
            'node_update.update_node_types',
            ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']
        )
        
        # Get operational filter from config
        self.only_operational = self.config.get(
            'node_update.only_operational', True
        )
        
        self.running = False
        self.stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_update_time': None
        }
        
        # Setup logging from config
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging based on config"""
        log_level = self.config.get('scheduler.log_level', 'INFO')
        log_file = self.config.get('scheduler.log_file', 'node_update.log')
        enable_logging = self.config.get('scheduler.enable_logging', True)
        
        if enable_logging:
            # Update logging level
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            logger.setLevel(numeric_level)
            
            # Add file handler if not already added
            if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                logger.addHandler(file_handler)
    
    def start(self):
        """Start the scheduler"""
        self.running = True
        logger.info(f"Node Position Scheduler started (interval: {self.update_interval}s)")
        
        try:
            while self.running:
                start_time = time.time()
                
                try:
                    self._update_all_positions()
                    self.stats['successful_updates'] += 1
                except Exception as e:
                    logger.error(f"Error updating positions: {str(e)}", exc_info=True)
                    self.stats['failed_updates'] += 1
                
                self.stats['total_updates'] += 1
                self.stats['last_update_time'] = datetime.now().isoformat()
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Update took {elapsed:.2f}s, longer than interval {self.update_interval}s")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping scheduler...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Node Position Scheduler stopped")
        self._print_stats()
    
    def _update_all_positions(self):
        """Update positions of all satellites"""
        if not db.is_connected():
            logger.warning("Database not connected, attempting to reconnect...")
            db.connect()
        
        if not db.is_connected():
            raise Exception("Cannot connect to database")
        
        nodes_collection = db.get_collection('nodes')
        if nodes_collection is None:
            raise Exception("Cannot get nodes collection")
        
        current_time = time.time()
        
        # Build query from config
        query = {
            'nodeType': {'$in': self.update_node_types}
        }
        
        if self.only_operational:
            query['isOperational'] = True
        
        # Get all satellites
        satellites = list(nodes_collection.find(query))
        
        if not satellites:
            logger.debug("No satellites found to update")
            return
        
        updated_count = 0
        
        for satellite in satellites:
            try:
                # Calculate new position
                new_position = self.orbital_engine.calculate_position_at_time(
                    satellite, current_time
                )
                
                # Calculate velocity
                new_velocity = self.orbital_engine.calculate_velocity(
                    satellite, current_time
                )
                
                # Update in database
                update_result = nodes_collection.update_one(
                    {'nodeId': satellite['nodeId']},
                    {
                        '$set': {
                            'position': new_position,
                            'velocity': new_velocity,
                            'lastPositionUpdate': datetime.now().isoformat(),
                            'positionTimestamp': current_time,
                            'lastUpdated': datetime.now().isoformat()
                        }
                    }
                )
                
                if update_result.modified_count > 0:
                    updated_count += 1
                    logger.debug(
                        f"Updated {satellite['nodeId']}: "
                        f"lat={new_position['latitude']:.2f}, "
                        f"lon={new_position['longitude']:.2f}, "
                        f"alt={new_position['altitude']/1000:.2f}km"
                    )
            
            except Exception as e:
                logger.error(f"Error updating satellite {satellite.get('nodeId', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Updated {updated_count}/{len(satellites)} satellite positions")
    
    def _print_stats(self):
        """Print statistics"""
        logger.info("=" * 50)
        logger.info("Scheduler Statistics:")
        logger.info(f"  Total updates: {self.stats['total_updates']}")
        logger.info(f"  Successful: {self.stats['successful_updates']}")
        logger.info(f"  Failed: {self.stats['failed_updates']}")
        logger.info(f"  Last update: {self.stats['last_update_time']}")
        logger.info("=" * 50)
    
    def update_once(self):
        """Run a single update (for testing)"""
        logger.info("Running single update...")
        self._update_all_positions()
        self._print_stats()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Scheduled Node Position Update Service'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Update interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run a single update and exit'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon process (not implemented yet)'
    )
    
    args = parser.parse_args()
    
    # Validate interval
    if args.interval < 1:
        logger.error("Interval must be at least 1 second")
        sys.exit(1)
    
    # Create scheduler
    scheduler = NodePositionScheduler(update_interval=args.interval)
    
    if args.once:
        # Run once and exit
        scheduler.update_once()
    else:
        # Run continuously
        try:
            scheduler.start()
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}", exc_info=True)
            sys.exit(1)

if __name__ == '__main__':
    main()


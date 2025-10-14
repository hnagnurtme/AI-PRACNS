# main.py
import os
import glob
import time
from utils.config import TrainingConfig
from data.mongo_manager import MongoDataManager
from training.training_manager import TrainingManager

def find_latest_checkpoint():
    """Tìm checkpoint mới nhất với improved pattern matching"""
    checkpoint_patterns = [
        "models/latest_checkpoint.pth",
        "models/checkpoint_*.pth", 
        "models/best_model_*.pth",
        "models/training_*.pth",
        "saved_models/*.pth"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))
    
    if checkpoints:
        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        print(f"🔍 Found {len(checkpoints)} checkpoints")
        return checkpoints[0]
    
    return None

def list_available_checkpoints():
    """List tất cả checkpoints available với thông tin chi tiết"""
    checkpoint_patterns = ["models/*.pth", "saved_models/*.pth"]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))
    
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints
    
    return []

def setup_directories():
    """Tạo các thư mục cần thiết"""
    directories = ["models", "logs", "saved_models", "results"]
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def validate_environment():
    """Validate môi trường trước khi training"""
    print("🔍 Validating environment...")
    
    # Check required directories
    if not os.path.exists("models"):
        print("⚠️  Creating 'models' directory...")
        os.makedirs("models", exist_ok=True)
    
    # Check if we can write to models directory
    try:
        test_file = "models/test_write.permission"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ Write permissions: OK")
    except Exception as e:
        print(f"❌ Cannot write to models directory: {e}")
        return False
    
    return True

def print_training_summary(config, node_count, checkpoint_info):
    """In summary trước khi bắt đầu training"""
    print("\n" + "="*60)
    print("🎯 TRAINING SUMMARY")
    print("="*60)
    print(f"📊 Network Stats:")
    print(f"   • Total Nodes: {node_count}")
    print(f"   • Training Episodes: {config.total_episodes:,}")
    print(f"   • Batch Size: {config.batch_size}")
    print(f"   • Learning Rate: {config.learning_rate}")
    
    print(f"⚙️  Training Config:")
    print(f"   • Gamma (discount): {config.gamma}")
    print(f"   • Warmup Steps: {config.warmup_steps}")
    print(f"   • Target Update Freq: {config.target_update_freq}")
    
    if checkpoint_info['found']:
        print(f"🔄 Resume Info:")
        print(f"   • Checkpoint: {checkpoint_info['path']}")
        print(f"   • Episode: {checkpoint_info.get('episode', 'Unknown')}")
        print(f"   • Best Reward: {checkpoint_info.get('best_reward', 'Unknown')}")
    else:
        print(f"🆕 Starting: Fresh Training")
    
    print("="*60)

def main():
    print("🚀 SAGINs DRL Routing Training System")
    print("=" * 50)
    
    # Setup environment
    setup_directories()
    if not validate_environment():
        print("❌ Environment validation failed!")
        return
    
    # Enhanced Configuration với hyperparameters được tối ưu
    config = TrainingConfig(
        total_episodes=10000,        # Giảm xuống để training nhanh hơn
        warmup_steps=1000,           # Giảm warmup
        batch_size=64,               # Batch size phù hợp
        learning_rate=1e-4,          # Learning rate được điều chỉnh
        target_update_freq=500,      # Cập nhật target network
        gamma=0.98,                  # Discount factor
        epsilon_start=0.3,
        epsilon_end=0.02,
        epsilon_decay=0.997
    )
    
    # MongoDB connection với error handling
    try:
        mongo_manager = MongoDataManager(
            host=config.mongo_host,
            port=config.mongo_port,
            db_name=config.db_name,
            username=config.db_username,
            password=config.db_password,
            auth_source=config.db_auth_source
        )
        
        # Test connection với timeout
        print("🔗 Testing MongoDB connection...")
        snapshot = mongo_manager.get_training_snapshot()
        nodes = snapshot.get('nodes', {})
        
        if not nodes:
            print("❌ No nodes found in database!")
            return
            
        print(f"✅ MongoDB Connected - Nodes: {len(nodes)}")
        
        # Show node types chi tiết
        node_types = {}
        node_status = {'healthy': 0, 'unhealthy': 0}
        
        for node_id, node_data in nodes.items():
            node_type = node_data.get('nodeType', 'UNKNOWN')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_data.get('healthy', True) and node_data.get('isOperational', True):
                node_status['healthy'] += 1
            else:
                node_status['unhealthy'] += 1
        
        print(f"📊 Node Analysis:")
        for node_type, count in node_types.items():
            print(f"   • {node_type}: {count}")
        print(f"   • Healthy: {node_status['healthy']}, Unhealthy: {node_status['unhealthy']}")
            
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        print("💡 Please check:")
        print("   - MongoDB is running")
        print("   - Connection string is correct") 
        print("   - Authentication credentials are valid")
        return
    
    # Checkpoint handling với improved logic
    checkpoint_info = {'found': False, 'path': None, 'episode': 0, 'best_reward': 0}
    
    all_checkpoints = list_available_checkpoints()
    
    if all_checkpoints:
        print(f"\n📁 Found {len(all_checkpoints)} Checkpoints:")
        for i, checkpoint in enumerate(all_checkpoints[:5]):
            mtime = os.path.getmtime(checkpoint)
            size_mb = os.path.getsize(checkpoint) / (1024*1024)
            modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            
            # Try to extract episode info từ filename
            episode_num = "Unknown"
            if 'ep' in checkpoint:
                try:
                    import re
                    ep_match = re.search(r'ep(\d+)', checkpoint)
                    if ep_match:
                        episode_num = f"{int(ep_match.group(1)):,}"
                except:
                    pass
                    
            print(f"   {i+1}. {os.path.basename(checkpoint)}")
            print(f"      Episode: {episode_num} | Size: {size_mb:.1f}MB | Modified: {modified_time}")
        
        if len(all_checkpoints) > 5:
            print(f"   ... and {len(all_checkpoints) - 5} more")
        
        # User choice với validation
        print(f"\n🎯 Training Options:")
        print(f"   1. Resume from latest checkpoint")
        print(f"   2. Choose specific checkpoint") 
        print(f"   3. Start fresh training")
        print(f"   4. Exit")
        
        while True:
            choice = input("Choose option (1-4): ").strip()
            
            if choice == '1':
                selected_checkpoint = all_checkpoints[0]
                checkpoint_info.update({'found': True, 'path': selected_checkpoint})
                print(f"🔄 Resuming from: {os.path.basename(selected_checkpoint)}")
                break
                
            elif choice == '2':
                print(f"\nSelect checkpoint (1-{min(10, len(all_checkpoints))}):")
                for i, checkpoint in enumerate(all_checkpoints[:10]):
                    print(f"   {i+1}. {os.path.basename(checkpoint)}")
                
                try:
                    selection = int(input("Enter number: ")) - 1
                    if 0 <= selection < len(all_checkpoints):
                        selected_checkpoint = all_checkpoints[selection]
                        checkpoint_info.update({'found': True, 'path': selected_checkpoint})
                        print(f"🔄 Using: {os.path.basename(selected_checkpoint)}")
                        break
                    else:
                        print("❌ Invalid selection")
                except ValueError:
                    print("❌ Please enter a valid number")
                    
            elif choice == '3':
                print("🆕 Starting fresh training...")
                break
                
            elif choice == '4':
                print("👋 Exiting...")
                return
            else:
                print("❌ Please enter 1, 2, 3, or 4")
    
    else:
        print("\n💡 No existing checkpoints found. Starting fresh training...")
    
    # Hiển thị training summary
    print_training_summary(config, len(nodes), checkpoint_info)
    
    # Final confirmation
    print("\n⚠️  Ready to start training!")
    confirm = input("Press Enter to start, or 'q' to quit: ").strip().lower()
    if confirm == 'q':
        print("👋 Training cancelled.")
        return
    
    # Initialize Training Manager
    try:
        print("\n🎬 Initializing Training Manager...")
        trainer = TrainingManager(
            config=config, 
            mongo_manager=mongo_manager,
            resume_from_checkpoint=checkpoint_info['path'] if checkpoint_info['found'] else None
        )
        
        # Start training với timestamp
        start_time = time.time()
        print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        trainer.start_training()
        
        # Training completed
        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n🏁 Training completed!")
        print(f"⏱️  Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"📈 Final Episode: {trainer.episode}")
        print(f"🎯 Best Average Reward: {trainer.best_avg_reward:.2f}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user at episode {getattr(trainer, 'episode', 'Unknown')}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
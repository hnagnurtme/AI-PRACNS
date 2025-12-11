#!/bin/bash

# Retrain script with improved reward function
# Usage: ./scripts/retrain.sh [episodes]

set -e

EPISODES=${1:-2000}
BACKEND_DIR="/Users/anhnon/AIPRANCS/Backend"

echo "=========================================="
echo "🔄 Retraining RL Model with Improved Rewards"
echo "=========================================="
echo ""
echo "Episodes: $EPISODES"
echo "Backend: $BACKEND_DIR"
echo ""

# Backup old models
echo "📦 Backing up old models..."
BACKUP_DIR="$BACKEND_DIR/models/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "$BACKEND_DIR/models/best_models/best_model.pt" ]; then
    mv "$BACKEND_DIR/models/best_models/best_model.pt" "$BACKUP_DIR/"
    echo "   ✓ Backed up best_model.pt"
fi

if [ -f "$BACKEND_DIR/models/rl_agent/final_model.pt" ]; then
    mv "$BACKEND_DIR/models/rl_agent/final_model.pt" "$BACKUP_DIR/"
    echo "   ✓ Backed up final_model.pt"
fi

if [ -d "$BACKEND_DIR/models/checkpoints" ]; then
    cp -r "$BACKEND_DIR/models/checkpoints" "$BACKUP_DIR/"
    rm -rf "$BACKEND_DIR/models/checkpoints"
    mkdir -p "$BACKEND_DIR/models/checkpoints"
    echo "   ✓ Backed up and cleared checkpoints"
fi

echo ""
echo "🚀 Starting training..."
echo "   - Max steps per episode: 10 (was 15)"
echo "   - Step penalty: -8.0 (was -5.0)"
echo "   - Hop penalty: -12.0 (was -8.0)"
echo "   - Max optimal hops: 6 (was 10)"
echo "   - Success distance threshold: 500-1000km (was 2000km)"
echo ""

cd "$BACKEND_DIR"
python training/train.py --episodes $EPISODES

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "📊 Next steps:"
echo "   1. Check training logs for average hops (should be ~4-5)"
echo "   2. Run validation: jupyter notebook notebooks/002_allocation_comparison.ipynb"
echo "   3. Compare with Dijkstra baseline"
echo ""
echo "💾 Old models backed up to: $BACKUP_DIR"
echo ""

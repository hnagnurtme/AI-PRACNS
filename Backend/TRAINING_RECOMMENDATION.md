# Training Recommendation & Docker Commands

## 📊 Test Results Analysis

### Current Test Results (GS-to-GS Routing):
- ✅ **RL Success Rate**: 100.0%
- ✅ **Dijkstra Success Rate**: 100.0%
- ✅ **RL Average Hops**: 3.10 (vs Dijkstra: 4.23)
- ✅ **RL finds shorter paths** (fewer hops)

### Conclusion:
Model **successfully learns GS-to-GS routing** and performs better than Dijkstra!

---

## 🔄 Should You Retrain?

### ✅ **YES, Recommended to Retrain** because:

1. **Code Changes**: Trainer has been modified to:
   - Select ground stations for terminals BEFORE training
   - Only learn routing between GS (not terminal-to-GS selection)
   - Use explicit `source_ground_station` and `dest_ground_station` in environment

2. **Model Architecture**: Current model was trained with old logic that may have learned:
   - Terminal-to-GS selection (which should be separate algorithm)
   - Mixed objectives (routing + selection)

3. **Better Focus**: New training will:
   - Focus purely on GS-to-GS routing optimization
   - Learn more efficient paths between ground stations
   - Potentially improve performance even further

### ⚠️ **However**, if:
- Current model works well for your use case
- You don't have time/resources for retraining
- You want to test current model more first

**Then you can skip retraining for now**, but it's recommended to retrain when possible.

---

## 🐳 Docker Commands

### Restart Backend Service

```bash
# Quick restart (recommended)
docker compose restart backend

# Or from project root
cd /Users/anhnon/AI-PRACNS
docker compose restart backend
```

### Other Useful Commands

```bash
# View backend logs
docker compose logs -f backend

# Rebuild and restart (if code changed)
docker compose up -d --build backend

# Stop backend
docker compose stop backend

# Start backend
docker compose start backend

# Check backend status
docker compose ps backend

# Check backend health
curl http://localhost:8080/api/health
```

### Full Service Management

```bash
# Restart all services
docker compose restart

# Stop all services
docker compose down

# Start all services
docker compose up -d

# View all logs
docker compose logs -f
```

---

## 🚀 Training Commands

### If You Decide to Retrain:

```bash
# Navigate to Backend directory
cd /Users/anhnon/AI-PRACNS/Backend

# Option 1: Standard training
python -m training.train

# Option 2: Enhanced training (with curriculum & imitation learning)
# Make sure config.dev.yaml has:
#   training:
#     use_enhanced_trainer: true
python -m training.train

# Option 3: Custom number of episodes
python -m training.train --episodes 2000

# Option 4: With custom config
python -m training.train --config config.dev.yaml
```

### Training Configuration

Make sure `config.dev.yaml` has:
```yaml
training:
  use_enhanced_trainer: true  # Use Enhanced Trainer
  max_episodes: 2000
  max_steps_per_episode: 15

curriculum:
  enabled: true

imitation_learning:
  enabled: true
```

---

## 📝 Summary

1. **Test Results**: ✅ Excellent (100% success, better than Dijkstra)
2. **Retraining**: ✅ **Recommended** (to learn pure GS-to-GS routing)
3. **Docker Restart**: `docker compose restart backend`
4. **Training Command**: `python -m training.train`

---

## 🎯 Next Steps

1. **Restart backend** to apply code changes:
   ```bash
   docker compose restart backend
   ```

2. **Optional: Retrain model** for better GS-to-GS routing:
   ```bash
   cd Backend
   python -m training.train
   ```

3. **Test again** with new model (if retrained):
   - Run `012_test_gs_to_gs_routing.ipynb` again
   - Compare results with previous test


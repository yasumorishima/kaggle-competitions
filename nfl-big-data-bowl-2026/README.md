# NFL Big Data Bowl 2026

Player trajectory prediction using geometric rules and physics-based approach.

## 📋 Competition Overview

- **Host:** NFL (National Football League)
- **Platform:** Kaggle
- **Task:** Predict player positions in future frames after ball is thrown
- **Type:** Time series trajectory prediction

## 🏆 Achievement

**🥉 Bronze Medal (Notebook)**

- **Public Leaderboard:** 2.921 yards RMSE
- **Execution Time:** <5 seconds
- **Recognition:** Community-recognized notebook contribution

**Note:** This medal was earned through notebook sharing and community votes, not competition placement.

## 🎯 Approach

### Core Philosophy: Simplicity over Complexity

Instead of complex machine learning models, this solution uses **physics-based geometric rules** that capture essential player movement patterns.

### Key Insights

#### 1. Targeted Receivers → Direct Path to Ball
```python
# Receivers know where ball will land
end_position = ball_landing_point
```
Players with the ball thrown to them run directly toward the ball landing point.

#### 2. Defensive Coverage → Mirror Receivers
```python
# Defenders maintain spatial relationship
defender_position = receiver_position + distance_based_offset
```
Defenders mirror the receiver they're guarding with distance-based offset adjustment:
- **Tight coverage** (<5 yards): 0.8x offset
- **Medium coverage** (5-10 yards): 0.6x offset  
- **Zone coverage** (>10 yards): 0.4x offset

#### 3. Linear Interpolation
```python
# Smooth movement across frames
position(t) = start + progress * (end - start)
```
Short time horizons make linear approximation effective.

## 💡 Why This Works

### Domain Knowledge > Complex Models

```
Simple geometric rules capture essential patterns:
✓ Receivers know ball destination → direct path
✓ Defenders track receivers → maintain relationship
✓ Short prediction window → linear approximation valid
✓ Field boundaries → natural constraints
```

### Performance vs Complexity

| Metric | Value |
|--------|-------|
| RMSE | 2.921 yards |
| Training Time | 0 seconds (rule-based) |
| Inference Time | <5 seconds |
| Code Lines | ~150 (clean & readable) |

## 🔍 Technical Details

### Step-by-Step Process

1. **Velocity Calculation**
   ```python
   vx = speed * cos(direction)
   vy = speed * sin(direction)
   ```

2. **Default Endpoint (Momentum-based)**
   ```python
   end_x = start_x + vx * time_horizon
   end_y = start_y + vy * time_horizon
   ```

3. **Override for Targeted Receivers**
   ```python
   if player_role == 'Targeted Receiver':
       end_x = ball_land_x
       end_y = ball_land_y
   ```

4. **Defensive Coverage Adjustment**
   ```python
   offset_x = defender_x - receiver_x
   offset_y = defender_y - receiver_y
   offset_factor = f(initial_distance)
   
   defender_end_x = receiver_end_x + offset_x * offset_factor
   ```

5. **Linear Interpolation**
   ```python
   progress = (frame_id - 1) / num_frames
   x(t) = start_x + progress * (end_x - start_x)
   ```

6. **Field Constraints**
   ```python
   x = clip(x, 0, 120)    # Field length
   y = clip(y, 0, 53.3)   # Field width
   ```

## 🛠️ Tech Stack

- **Language:** Python 3
- **Core Libraries:** pandas, polars, numpy
- **Approach:** Rule-based (no ML training)
- **Environment:** Kaggle Notebooks

## 📊 Results Analysis

### Strengths
- ✅ Fast execution (<5 seconds)
- ✅ No training required
- ✅ Interpretable predictions
- ✅ Competitive baseline performance
- ✅ Robust to edge cases

### Limitations
- ❌ Assumes linear movement (ignores acceleration)
- ❌ Single receiver focus (doesn't handle coverage switches)
- ❌ No field zone consideration (red zone differs)
- ❌ Speed not factored into trajectory shape

## 🎓 Key Learnings

### 1. Domain Knowledge is Powerful
Understanding football dynamics allowed creation of effective rules without data-driven training.

### 2. Simple Baselines First
Before jumping to deep learning, establish what simple approaches can achieve.

### 3. Interpretability Matters
Rule-based predictions are easy to debug and explain.

### 4. Physics Constraints Help
Field boundaries and player roles provide natural constraints that improve predictions.

## 🚀 Future Improvements

If extending this approach:

1. **Acceleration Modeling**
   ```python
   # Consider acceleration in trajectory
   x(t) = x0 + v0*t + 0.5*a*t^2
   ```

2. **Field Zone Context**
   ```python
   # Red zone behavior differs
   if field_position < 20:
       adjust_strategy()
   ```

3. **Multiple Receiver Handling**
   ```python
   # Defenders may switch coverage
   nearest_receiver = find_closest(defenders, receivers)
   ```

4. **Speed-Based Trajectory**
   ```python
   # Faster players → more curved paths
   curvature = f(player_speed)
   ```

## 📁 Files

```
nfl-big-data-bowl-2026/
├── geometric-rules-baseline.ipynb    # Main notebook (Bronze Medal)
└── README.md                          # This file
```

## 🔗 Links

- **Competition:** [Kaggle Competition Page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026)
- **Notebook:** [Geometric Rules Baseline - 2.921 RMSE](https://www.kaggle.com/code/yasunorim/geometric-rules-baseline-2-921-rmse-no-ml)

## 📝 Notes

This was my first sports analytics competition. The experience taught me:
- Value of domain knowledge in feature engineering
- Importance of establishing strong baselines
- Trade-offs between complexity and interpretability
- How geometric constraints can replace learned patterns

**Achievement Type:** This Bronze Medal was earned through notebook sharing and community recognition, demonstrating the educational value and code quality rather than competition ranking.

---

*Community-recognized notebook. Not a competition placement medal.*

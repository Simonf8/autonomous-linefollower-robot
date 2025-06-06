# 🤖 Enhanced Line Follower Robot with Advanced Obstacle Detection

An upgraded version of the autonomous line following robot with **advanced obstacle detection and avoidance capabilities**. This system now includes multiple detection methods including optional YOLO integration for superior object recognition.

## 🌟 New Enhanced Features

### 🔍 Advanced Obstacle Detection
- **Multiple Detection Methods**: Combines contour analysis, motion detection, and optional YOLO
- **YOLO Integration**: YOLOv5 nano model for real-time object classification
- **Graceful Fallback**: Works without YOLO using enhanced contour detection
- **Motion Detection**: Detects moving objects using background subtraction
- **Smart Filtering**: Removes false positives and duplicates

### 🎯 Intelligent Avoidance System
- **Threat Assessment**: Prioritizes obstacles based on position and size
- **Avoidance Cooldown**: Prevents oscillating behavior
- **Directional Logic**: Smart left/right avoidance decisions
- **Recovery Mechanism**: Returns to line following after avoidance

### 📊 Enhanced Line Detection
- **Multi-Zone Analysis**: Bottom zone for line following, middle for prediction, top for obstacles
- **Improved Filtering**: Better line shape recognition with aspect ratio analysis
- **Confidence Scoring**: More reliable line detection with quality metrics
- **Predictive Tracking**: Uses middle zone to predict upcoming turns

## 🚀 Quick Start

### 1. Installation

```bash
# Run the enhanced setup script
./setup_enhanced.sh

# Or manual installation:
pip3 install -r requirements_lite.txt

# Optional: Install YOLO for advanced detection
pip3 install ultralytics torch torchvision
```

### 2. Configuration

Update the ESP32 IP address in the script:
```python
ESP32_IP = '192.168.53.117'  # Change to your ESP32's IP
```

### 3. Run the Enhanced System

```bash
python3 tests/integration/test_enhanced_simple.py
```

### 4. Access Web Dashboard

Open your browser to: `http://your_pi_ip:5000`

## 🔧 Detection System Overview

### Detection Zones

The enhanced system divides the camera view into three intelligent zones:

1. **🎯 Line Detection Zone (Bottom 35%)**
   - Primary line following
   - High-confidence line detection
   - Real-time steering calculations

2. **🔮 Prediction Zone (Middle 25%)**
   - Corner prediction
   - Upcoming turn detection
   - Smooth trajectory planning

3. **⚠️ Obstacle Detection Zone (Top 40%)**
   - Object detection and classification
   - Threat assessment
   - Avoidance path planning

### Detection Methods

#### 1. YOLO Detection (Optional)
```python
# Automatically detects:
- People, vehicles, animals
- Furniture and objects
- Accurate bounding boxes
- Confidence scores
```

#### 2. Enhanced Contour Detection
```python
# Improved features:
- Bilateral filtering for noise reduction
- Adaptive thresholding for varying light
- Morphological operations for cleanup
- Shape analysis for better filtering
```

#### 3. Motion Detection
```python
# Detects moving obstacles:
- Background subtraction
- Frame differencing
- Moving object tracking
- Dynamic threat assessment
```

## ⚙️ Configuration Parameters

### Obstacle Detection Settings
```python
OBSTACLE_MIN_AREA = 1500          # Minimum obstacle size
OBSTACLE_MIN_WIDTH_RATIO = 0.15   # Minimum width to block path
OBSTACLE_MAX_DISTANCE_RATIO = 0.4 # Maximum distance from center
AVOIDANCE_DURATION = 10           # Frames to perform avoidance
AVOIDANCE_COOLDOWN = 20           # Cooldown between avoidances
```

### Enhanced PID Settings
```python
KP = 0.8    # More responsive proportional gain
KI = 0.03   # Improved integral gain
KD = 0.15   # Enhanced derivative gain
```

### Detection Zones
```python
ZONE_BOTTOM_HEIGHT = 0.35  # Line detection zone
ZONE_MIDDLE_HEIGHT = 0.25  # Prediction zone  
ZONE_TOP_HEIGHT = 0.40     # Obstacle detection zone
```

## 📊 Web Dashboard Features

### Enhanced Status Display
- Real-time obstacle count and threat level
- Detection method indicators (YOLO/Enhanced/Motion)
- Avoidance maneuver statistics
- Performance metrics and FPS

### Visual Indicators
- **🟢 Green**: Normal line following
- **🟡 Orange**: Obstacles detected but manageable
- **🔴 Red**: Active threat avoidance
- **⚪ Gray**: Line lost/searching

### Detection Method Status
```
YOLO Detection: Available/Fallback Mode
Contour Analysis: Enhanced
Motion Detection: Active
```

## 🛡️ Safety Features

### Obstacle Avoidance Logic
1. **Threat Detection**: Identifies objects in robot's path
2. **Direction Decision**: Chooses optimal avoidance direction
3. **Execution**: Performs avoidance maneuver
4. **Cooldown**: Prevents immediate re-triggering
5. **Recovery**: Returns to line following

### False Positive Filtering
- Size and shape validation
- Position relevance checking
- Confidence thresholding
- Duplicate removal

## 🔧 Troubleshooting

### Common Issues

#### YOLO Not Loading
```bash
# Check if YOLO is properly installed
python3 -c "from ultralytics import YOLO; print('YOLO OK')"

# If failed, reinstall:
pip3 install ultralytics torch torchvision
```

#### Too Many False Detections
```python
# Increase confidence thresholds:
OBSTACLE_CONFIDENCE_THRESHOLD = 0.7  # Increase from 0.6
OBSTACLE_MIN_AREA = 2000             # Increase from 1500
```

#### Robot Too Aggressive in Avoidance
```python
# Reduce avoidance sensitivity:
OBSTACLE_MAX_DISTANCE_RATIO = 0.3    # Reduce from 0.4
AVOIDANCE_DURATION = 8               # Reduce from 10
```

#### Poor Line Detection
```python
# Adjust line detection parameters:
BLACK_THRESHOLD = 70                 # Increase from 60
MIN_CONTOUR_AREA = 200              # Increase from 150
```

## 📈 Performance Optimization

### For Raspberry Pi 4
- Full YOLO detection enabled
- All enhancement features active
- 15-20 FPS typical performance

### For Raspberry Pi 3B+
- Enhanced contour + motion detection
- Optional YOLO (may reduce FPS)
- 10-15 FPS typical performance

### Memory Usage
- Basic enhanced: ~150MB RAM
- With YOLO: ~300-400MB RAM

## 🔄 Upgrade Path

### From Basic to Enhanced
1. Run the enhanced setup script
2. Update ESP32 IP configuration
3. Test with basic enhancements first
4. Optionally add YOLO detection

### Adding YOLO Later
```bash
# Install YOLO dependencies
pip3 install ultralytics torch torchvision

# Restart the robot script
# YOLO will be automatically detected and enabled
```

## 📊 Performance Comparison

| Feature | Basic | Enhanced | Enhanced + YOLO |
|---------|-------|----------|-----------------|
| Obstacle Detection | Simple contours | Multi-method | AI-powered |
| False Positives | High | Low | Very Low |
| Object Recognition | None | Shape-based | Classification |
| Avoidance Logic | Basic | Smart | Intelligent |
| FPS (Pi 4) | 20-25 | 15-20 | 12-18 |
| Memory Usage | 100MB | 150MB | 350MB |

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Additional detection algorithms
- Better avoidance strategies
- Performance optimizations
- New sensor integrations

## 📜 License

This enhanced version maintains the same license as the original project.

---

**Happy robot building! 🤖✨** 
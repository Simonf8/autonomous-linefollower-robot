# Autonomous Line-Following Robot 🤖

A sophisticated autonomous robot system featuring computer vision, A\* pathfinding, and hybrid localization. This project integrates high-level mission planning on a Raspberry Pi with low-level motor control on an ESP32 (or direct GPIO control).

## Key Features

- **Advanced Computer Vision**:
  - Robust line following with look-ahead strategy.
  - Obstacle detection and avoidance using edge detection and YOLO.
  - Object detection for mission-specific targets (e.g., boxes for pickup).
- **Intelligent Navigation**:
  - A\* pathfinding algorithm with configurable turn penalties for optimized routing.
  - Maze navigation using a pre-defined grid map.
- **Hybrid Localization**:
  - Fusion of wheel encoders and visual markers for precise position tracking.
  - Dead reckoning with camera-based drift correction.
- **Dynamic Control System**:
  - PID-based steering for smooth line following.
  - Support for multi-mode cornering: smooth, sideways (omni), pivot, and front-turn.
- **Web-Based Command Center**:
  - Real-time Flask UI for live video streams, telemetry, and manual control.
  - Teleoperation support for testing and mission overrides.

## Repository Structure

```text
.
├── src/
│   ├── controllers/         # Core logic and vision processing
│   │   ├── main.py          # Main entry point and state machine
│   │   ├── config.py        # Project-wide configuration parameters
│   │   ├── camera_line_follower.py # CV-based line detection
│   │   ├── pathfinder.py    # A* path planning logic
│   │   ├── pid.py           # Consolidated PID controllers
│   │   ├── pi_motor_controller.py # Hardware motor interface
│   │   └── ...              # Other specialized modules
│   └── esp32/               # Firmware for ESP32 motor control
├── models/                  # ML models (YOLO weights)
├── static/ & templates/     # Web Command Center assets
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- **Hardware**: Raspberry Pi 4/5 (recommended) or ESP32-S3.
- **OS**: Raspberry Pi OS (64-bit) or Ubuntu.
- **Python**: 3.9+

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Simonf8/autonomous-linefollower-robot.git
   cd autonomous-linefollower-robot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

To start the main controller and web interface:

```bash
python src/controllers/main.py
```

Then open `http://<robot-ip>:5000` in your browser.

## ⚙️ Configuration

Mission parameters, PID constants, and hardware pins can be tuned in `src/controllers/config.py`.

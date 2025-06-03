# 🤖 Autonomous Line Following Robot

A sophisticated 4-wheel line following robot using **Raspberry Pi** for computer vision and **ESP32** (MicroPython) for motor control with encoders.

## 🌟 Features

- **Advanced Computer Vision**: Enhanced line detection with median filtering and grid-based analysis
- **4-Wheel Drive**: Differential steering with encoder feedback
- **Real-time Web Dashboard**: Monitor robot status, camera feed, and control parameters
- **WiFi Communication**: Raspberry Pi sends commands to ESP32 over WiFi
- **PID Control**: Smooth and responsive line following with enhanced PID controller
- **Multiple Detection Methods**: Traditional Hough lines + grid-based backup detection
- **Robust Error Handling**: Command timeouts, connection recovery, and failsafes

## 📁 Project Structure

```
autonomous-linefollower-robot-2/
├── tests/integration/test.py          # Main Raspberry Pi vision code
├── esp32_line_follower.py             # ESP32 MicroPython motor controller
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── esp32_line_follower.ino           # Arduino version (not used with MicroPython)
```

## 🔧 Hardware Requirements

### Raspberry Pi Setup
- **Raspberry Pi 4** (recommended) or Pi 3B+
- **USB Camera** or Pi Camera module
- **MicroSD Card** (32GB+ recommended)
- **Power Supply** (5V 3A)

### ESP32 Setup
- **ESP32 Development Board** (DevKit or similar)
- **2x L298N Motor Drivers** (or equivalent)
- **4x DC Motors** with encoders
- **12V Battery Pack** for motors
- **5V Power Supply** for logic
- **Jumper Wires** and breadboards
- **4 Wheels** and chassis

### Additional Components
- **Black electrical tape** for line track
- **White surface** for contrast
- **Common ground** connections between all components

## 🚀 Quick Start

### 1. Raspberry Pi Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip install -r requirements.txt

# Enable camera (if using Pi Camera)
sudo raspi-config
# Navigate to Interface Options > Camera > Enable

# Test camera
python3 -c "import cv2; print('Camera check:', cv2.VideoCapture(0).isOpened())"
```

### 2. ESP32 Setup

#### Install MicroPython on ESP32:

```bash
# Install esptool
pip install esptool

# Download MicroPython firmware
wget https://micropython.org/resources/firmware/esp32-20231005-v1.21.0.bin

# Flash firmware (replace /dev/ttyUSB0 with your ESP32 port)
esptool.py --chip esp32 --port /dev/ttyUSB0 erase_flash
esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash -z 0x1000 esp32-20231005-v1.21.0.bin
```

#### Upload Code:

1. **Using Thonny IDE** (Recommended):
   - Install Thonny: `sudo apt install thonny`
   - Open Thonny → Tools → Options → Interpreter
   - Select "MicroPython (ESP32)" and correct port
   - Open `esp32_line_follower.py`
   - Update WiFi credentials in the code
   - Save as `main.py` on the ESP32

2. **Using ampy**:
   ```bash
   pip install adafruit-ampy
   ampy --port /dev/ttyUSB0 put esp32_line_follower.py main.py
   ```

### 3. Hardware Wiring

#### Motor Driver Connections:

**Motor Driver 1 (Left Side)**:
```
ENA → GPIO 25    IN1 → GPIO 26    IN2 → GPIO 27
ENB → GPIO 14    IN3 → GPIO 12    IN4 → GPIO 13
```

**Motor Driver 2 (Right Side)**:
```
ENA → GPIO 32    IN1 → GPIO 33    IN2 → GPIO 25
ENB → GPIO 26    IN3 → GPIO 27    IN4 → GPIO 14
```

#### Encoder Connections:
```
Left Front:  A → GPIO 18, B → GPIO 19
Left Rear:   A → GPIO 21, B → GPIO 22
Right Front: A → GPIO 16, B → GPIO 17
Right Rear:  A → GPIO 4,  B → GPIO 2
```

#### Power Connections:
- **Motor Drivers**: 12V from battery, 5V logic power
- **ESP32**: 5V to VIN or 3.3V to 3V3
- **Common Ground**: Connect all ground pins together

### 4. Configuration

#### Update IP Address:
1. Power on ESP32 and check serial output for IP address
2. Update `ESP32_IP` in `tests/integration/test.py`:
   ```python
   ESP32_IP = '192.168.1.XXX'  # Replace with ESP32's IP
   ```

#### WiFi Credentials:
Update in `esp32_line_follower.py`:
```python
WIFI_SSID = "Your_WiFi_Network"
WIFI_PASSWORD = "Your_WiFi_Password"
```

### 5. Run the System

#### Start ESP32:
- Reset the ESP32 or power cycle
- Check serial monitor for successful WiFi connection
- Note the IP address displayed

#### Start Raspberry Pi:
```bash
cd /path/to/autonomous-linefollower-robot-2
python3 tests/integration/test.py
```

#### Access Web Dashboard:
Open browser and navigate to: `http://[PI_IP_ADDRESS]:5000`

## 🎛️ Web Dashboard

The web interface provides:
- **Live Camera Feed** with line detection overlay
- **Real-time Status** including confidence, offset, and steering
- **Motor Commands** and speed settings
- **Connection Status** to ESP32
- **Performance Metrics** (FPS, line detection confidence)

## ⚙️ Configuration Parameters

### Vision Parameters (in `test.py`):
```python
# Adjust these for your environment
BLACK_THRESHOLD = 80           # Line detection threshold
CANNY_LOW = 50                # Edge detection sensitivity
CANNY_HIGH = 150
PID_KP = 1.2                  # PID tuning parameters
PID_KI = 0.05
PID_KD = 0.35
```

### Motor Parameters (in `esp32_line_follower.py`):
```python
# Speed settings (0-1023 for 10-bit PWM)
SPEED_FAST = 1023
SPEED_NORMAL = 700
SPEED_SLOW = 400
SPEED_TURN = 600
```

## 🛠️ Troubleshooting

### Common Issues:

#### 1. No Camera Feed
```bash
# Check camera connection
lsusb  # Should show camera device
# Try different camera index
python3 -c "import cv2; cap=cv2.VideoCapture(1); print(cap.isOpened())"
```

#### 2. ESP32 Connection Failed
- Verify WiFi credentials
- Check IP address in Pi code
- Ensure both devices on same network
- Check ESP32 serial output for errors

#### 3. Motors Not Responding
- Verify power connections (12V to motor drivers)
- Check common ground connections
- Test motors directly with motor driver
- Verify GPIO pin connections

#### 4. Poor Line Detection
- Ensure good contrast (black line on white surface)
- Adjust `BLACK_THRESHOLD` value
- Check lighting conditions
- Verify camera focus

#### 5. Robot Oscillates on Line
- Reduce PID gains (start with `PID_KP = 0.5`)
- Increase `STEERING_DEADZONE`
- Check for mechanical issues

### Debug Mode:
Enable verbose logging by changing log level:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## 📈 Performance Optimization

### For Better Line Following:
1. **Lighting**: Use consistent, bright lighting
2. **Track**: High contrast black tape on white surface
3. **Camera**: Position for optimal line visibility
4. **PID Tuning**: Start conservative and gradually increase gains
5. **Speed**: Begin with slow speeds for tuning

### For Better Performance:
1. **Raspberry Pi**: Use Pi 4 for better processing power
2. **Camera**: Higher resolution helps with detection
3. **Network**: Use 5GHz WiFi for lower latency
4. **Memory**: Close unnecessary programs

## 🔄 System Architecture

```
┌─────────────────┐    WiFi Commands    ┌─────────────────┐
│   Raspberry Pi  │ ──────────────────► │      ESP32      │
│                 │                     │                 │
│ • Computer Vision│                     │ • Motor Control │
│ • Line Detection│                     │ • Encoder Read  │
│ • PID Control   │                     │ • WiFi Comm     │
│ • Web Dashboard │                     │ • Safety Checks │
│ • Command Gen   │                     │                 │
└─────────────────┘                     └─────────────────┘
         │                                       │
         │                                       │
    ┌────▼────┐                             ┌────▼────┐
    │ USB Cam │                             │ Motors  │
    │         │                             │ +       │
    └─────────┘                             │Encoders │
                                            └─────────┘
```

## 📝 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review ESP32 serial output
3. Check Raspberry Pi logs
4. Verify hardware connections

## 🎯 Future Enhancements

- [ ] Obstacle avoidance
- [ ] GPS waypoint navigation
- [ ] Multiple line following
- [ ] Machine learning integration
- [ ] Mobile app control
- [ ] Voice commands
- [ ] Remote monitoring

---

**Happy Robot Building! 🤖**


#!/bin/bash

echo "🤖 Enhanced Line Follower Robot Setup"
echo "====================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
echo "📋 Installing basic requirements..."
pip3 install -r requirements_lite.txt

echo "✅ Basic setup complete!"
echo ""
echo "🔍 Enhanced Obstacle Detection Options:"
echo "1. Test with basic enhanced detection (available now)"
echo "2. Install YOLO for advanced detection (requires more resources)"
echo ""

read -p "Do you want to install YOLO dependencies? (y/N): " install_yolo

if [[ $install_yolo =~ ^[Yy]$ ]]; then
    echo "🔄 Installing YOLO dependencies..."
    echo "⚠️  This may take several minutes and requires internet connection..."
    
    # Install PyTorch (CPU version for Raspberry Pi)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install Ultralytics YOLO
    pip3 install ultralytics
    
    echo "✅ YOLO dependencies installed!"
    echo "📝 Note: First run will download YOLOv5 model (~6MB)"
else
    echo "ℹ️  Skipping YOLO installation - using enhanced contour detection"
fi

echo ""
echo "🚀 Setup Instructions:"
echo "1. Update ESP32_IP in the script to match your ESP32"
echo "2. Make sure your ESP32 is running the line follower code"
echo "3. Run the enhanced robot: python3 tests/integration/test_enhanced_simple.py"
echo "4. Open web dashboard: http://your_pi_ip:5000"
echo ""
echo "🔧 Detection Features Available:"
echo "- ✅ Enhanced multi-zone line detection"
echo "- ✅ Improved contour-based obstacle detection"
echo "- ✅ Motion detection for moving obstacles"
echo "- ✅ Smart avoidance with cooldown system"
echo "- ✅ Background subtraction"
if [[ $install_yolo =~ ^[Yy]$ ]]; then
    echo "- ✅ YOLO-based object detection"
fi
echo ""
echo "✅ Enhanced Line Follower Robot is ready!" 
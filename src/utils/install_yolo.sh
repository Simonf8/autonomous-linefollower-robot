#!/bin/bash

echo "🚀 Installing YOLO11n for enhanced object detection..."

# Update pip
pip install --upgrade pip

# Install ultralytics (includes YOLO11n)
echo "📦 Installing ultralytics..."
pip install ultralytics

# Install additional dependencies if needed
echo "📦 Installing additional dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✅ YOLO11n installation complete!"
echo "🔄 Restart your robot script to use YOLO-powered object detection"
echo ""
echo "The robot will now:"
echo "  ✅ Accurately detect real objects (people, cups, bottles, etc.)"
echo "  ✅ Eliminate false positives from shadows and lighting"
echo "  ✅ Show object class names and confidence scores"
echo "  ✅ Much more reliable obstacle avoidance" 
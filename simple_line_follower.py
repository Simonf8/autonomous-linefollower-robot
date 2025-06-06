#!/usr/bin/env python3
"""
A simple, "from-scratch" line-following robot script.

Purpose:
This script provides the most basic, essential logic for a line follower.
It is designed to be easy to understand, debug, and tune. Get this working
first before moving to more complex versions.

How it works:
1.  Captures video from the camera.
2.  Focuses on a specific Region of Interest (ROI) at the bottom of the frame.
3.  Converts the image to black and white (binary thresholding).
4.  Finds the center (centroid) of the white line in the black-and-white image.
5.  Calculates the "error" - how far the line's center is from the center of the view.
6.  Uses a simple Proportional controller (P-controller) to calculate a steering correction.
7.  Sends a speed and turn command to the ESP32 over the network.
8.  Provides clear visual windows to show what the robot is doing, which is
    essential for debugging and tuning the THRESHOLD_VALUE.
"""
import cv2
import numpy as np
import socket
import time

# -----------------------------------------------------------------------------
# --- CONFIGURATION - TUNE THESE VALUES FOR YOUR ROBOT! ---
# -----------------------------------------------------------------------------

# -- Network Configuration --
# IMPORTANT: Change this to your ESP32's actual IP address
ESP32_IP = '192.168.53.117'
ESP32_PORT = 1234

# -- Camera Configuration --
CAM_INDEX = 0  # 0 is usually the default camera. Change if you have multiple cameras.
CAM_WIDTH = 320
CAM_HEIGHT = 240

# -- Line Following Logic Tuning --

# Region of Interest (ROI) - Percentage of the screen to focus on.
# We only care about the part of the line directly in front of the robot.
# These values are percentages of the camera height.
ROI_TOP = 0.60    # Start ROI at 60% from the top of the screen.
ROI_HEIGHT = 0.40 # Make the ROI 40% of the screen height.

# Thresholding - This is the MOST IMPORTANT parameter to tune.
# It determines what is considered "line" and what is "floor".
# The image is converted to grayscale (0=black, 255=white). Any pixel
# with a value BELOW this threshold will be considered part of the black line.
# ** HOW TO TUNE **: Look at the "Thresholded View" window. The goal is to
# have the black line appear as a solid white shape, and everything else
# should be black. Adjust this value until you get a clean result.
BLACK_LINE_THRESHOLD = 70

# Proportional-Controller (P-Controller) Gain.
# This single value determines how sharply the robot turns.
# - HIGHER Kp = Sharper, more responsive turns. Too high will cause oscillation.
# - LOWER Kp = Softer, smoother turns. Too low will cause the robot to lose the line on curves.
# ** HOW TO TUNE **: Start with a low value like 0.5. If the robot can't stay on
# the line during curves, increase it. If it wiggles back and forth too much,
# decrease it.
KP_GAIN = 1.0

# -- Robot Motion --
# Speed command to send to the ESP32. 'N' for Normal is a good start.
# This script uses a constant speed for simplicity.
MOTOR_SPEED_CMD = 'N'

# Steering Deadzone - If the steering correction is very small, just go straight.
# This prevents the robot from twitching when the line is almost centered.
STEERING_DEADZONE = 0.1 # +/- 10%

# -----------------------------------------------------------------------------
# --- ESP32 Communication Class ---
# -----------------------------------------------------------------------------

class ESP32Communicator:
    """Handles network communication with the ESP32."""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        """Attempts to connect to the ESP32."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0) # Set a timeout for connection attempts
            self.sock.connect((self.ip, self.port))
            print(f"✅ Successfully connected to ESP32 at {self.ip}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"❌ Failed to connect to ESP32: {e}")
            self.sock = None
            return False

    def send_command(self, speed, turn):
        """Sends a command to the ESP32. Reconnects if necessary."""
        if not self.sock:
            print("Not connected to ESP32. Attempting to reconnect...")
            if not self.connect():
                # If reconnection fails, wait a bit before trying again
                time.sleep(1)
                return False

        command = f"{speed}:{turn}\n"
        try:
            self.sock.sendall(command.encode('utf-8'))
            return True
        except (socket.error, socket.timeout) as e:
            print(f"💥 Lost connection to ESP32: {e}")
            self.sock.close()
            self.sock = None
            return False

    def close(self):
        """Sends a stop command and closes the connection."""
        if self.sock:
            print("Sending STOP command and closing connection.")
            try:
                self.send_command('H', 'FORWARD') # 'H' for Halt/Stop
            finally:
                self.sock.close()
                self.sock = None

# -----------------------------------------------------------------------------
# --- Main Application ---
# -----------------------------------------------------------------------------

def main():
    """The main function where the magic happens."""
    
    # --- Initialization ---
    print("🚀 Starting simple line following robot script...")
    
    # Connect to ESP32
    esp = ESP32Communicator(ESP32_IP, ESP32_PORT)

    # Initialize Camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"❌ FATAL: Cannot open camera at index {CAM_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print(f"📷 Camera opened successfully ({CAM_WIDTH}x{CAM_HEIGHT}).")
    
    # Calculate the ROI pixel dimensions
    roi_y_start = int(CAM_HEIGHT * ROI_TOP)
    roi_height = int(CAM_HEIGHT * ROI_HEIGHT)

    # --- Main Loop ---
    try:
        while True:
            # 1. CAPTURE FRAME
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Cannot read frame from camera. End of stream?")
                break
            
            # 2. CROP TO REGION OF INTEREST (ROI)
            roi = frame[roi_y_start : roi_y_start + roi_height, :]
            
            # 3. PROCESS IMAGE
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Apply blur to reduce noise
            blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            # Threshold the image. This is the key step.
            # cv2.THRESH_BINARY_INV makes the black line white and the floor black.
            _, thresholded_roi = cv2.threshold(blurred_roi, BLACK_LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

            # 4. FIND LINE CENTER
            # Calculate moments of the binary image
            M = cv2.moments(thresholded_roi)
            
            error = 0
            line_found = False

            if M["m00"] != 0:
                # A line was found, calculate its center (centroid)
                line_found = True
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 5. CALCULATE ERROR
                # Error is the distance from the line's center to the ROI's center
                roi_center_x = roi.shape[1] // 2
                error = cx - roi_center_x
            else:
                # No line was found in the ROI
                line_found = False

            # 6. CALCULATE STEERING & DETERMINE COMMANDS
            speed_cmd = 'H'  # Default to STOP
            turn_cmd = 'FORWARD'

            if line_found:
                # We see the line, so we can move
                speed_cmd = MOTOR_SPEED_CMD
                
                # Use the P-controller to get a steering correction value
                # We normalize the error to be between -1.0 and 1.0
                normalized_error = error / (roi.shape[1] / 2)
                steering_correction = KP_GAIN * normalized_error
                
                # Decide turn command based on the correction
                if steering_correction < -STEERING_DEADZONE:
                    turn_cmd = 'LEFT'
                elif steering_correction > STEERING_DEADZONE:
                    turn_cmd = 'RIGHT'
                else:
                    turn_cmd = 'FORWARD'
            else:
                # Line is lost, stop the robot for safety
                speed_cmd = 'H'
                turn_cmd = 'FORWARD'

            # 7. SEND COMMANDS TO ESP32
            esp.send_command(speed_cmd, turn_cmd)

            # 8. VISUALIZE FOR DEBUGGING
            # Draw the ROI rectangle on the main frame
            cv2.rectangle(frame, (0, roi_y_start), (CAM_WIDTH, roi_y_start + roi_height), (0, 255, 0), 2)
            
            # Draw info on the main frame
            status_text = f"Line Found: {line_found} | Error: {error}"
            command_text = f"Command: {speed_cmd}:{turn_cmd}"
            cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, command_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if line_found:
                # Draw the detected centroid on the ROI
                cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
                # Draw the center line of the ROI for reference
                cv2.line(roi, (roi_center_x, 0), (roi_center_x, roi.shape[0]), (255, 0, 0), 1)
            
            # Show the final images
            cv2.imshow("Line Follower View", frame)
            cv2.imshow("Thresholded View (Tune with this!)", thresholded_roi)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user.")
    finally:
        # --- Cleanup ---
        print("🧹 Cleaning up and stopping the robot...")
        esp.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done.")

if __name__ == '__main__':
    main()
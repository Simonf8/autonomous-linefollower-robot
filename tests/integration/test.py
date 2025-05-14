import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import sys

# ─── MOTOR SETUP ───────────────────────────────────────────────
IN1, IN2 = 21, 20   # Left motor (Forward: IN1=HIGH, IN2=LOW)
IN3, IN4 = 19, 16   # Right motor (Forward: IN3=HIGH, IN4=LOW)
PWM_FREQ = 500 # Reduced frequency slightly, 600 is fine too
SPEED = 60      # Default forward speed
TURN_SPEED = 70 # Speed for turning, can be higher for sharper turns
AVOID_SPEED = 50 # Speed for avoidance maneuvers

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in (IN1, IN2, IN3, IN4):
    GPIO.setup(pin, GPIO.OUT)

# Initialize PWM objects
# Left motor: p1=IN1, p2=IN2
# Right motor: p3=IN3, p4=IN4
p1 = GPIO.PWM(IN1, PWM_FREQ)
p2 = GPIO.PWM(IN2, PWM_FREQ)
p3 = GPIO.PWM(IN3, PWM_FREQ)
p4 = GPIO.PWM(IN4, PWM_FREQ)

for pwm in (p1, p2, p3, p4):
    pwm.start(0)

# ─── MOTOR CONTROL FUNCTIONS ──────────────────────────────────
def stop():
    p1.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(0)
    p3.ChangeDutyCycle(0)
    p4.ChangeDutyCycle(0)

def forward(speed=SPEED):
    p1.ChangeDutyCycle(speed)
    p2.ChangeDutyCycle(0)
    p3.ChangeDutyCycle(speed)
    p4.ChangeDutyCycle(0)

def reverse(speed=SPEED):
    p1.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(speed)
    p3.ChangeDutyCycle(0)
    p4.ChangeDutyCycle(speed)

def turn_left(speed=TURN_SPEED): # Pivot left
    p1.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(speed) # Left motor backward
    p3.ChangeDutyCycle(speed) # Right motor forward
    p4.ChangeDutyCycle(0)

def turn_right(speed=TURN_SPEED): # Pivot right
    p1.ChangeDutyCycle(speed) # Left motor forward
    p2.ChangeDutyCycle(0)
    p3.ChangeDutyCycle(0)     # Right motor backward
    p4.ChangeDutyCycle(speed)

# ─── ENCODER SETUP (Kept from original, not directly used in avoidance logic here) ───
ENC_A = 14
ENC_B = 15
pulse_count = 0
last_B = 0

def encoder_callback(channel):
    global pulse_count, last_B
    # Debounce or more robust logic might be needed for noisy encoders
    a_val = GPIO.input(ENC_A) # Read A state when B changes
    b_val = GPIO.input(ENC_B)
    if b_val != last_B: # Process on B change
        if b_val == 1: # B rising
            pulse_count += (1 if a_val == last_B else -1) # Check A for direction
        else: # B falling
            pulse_count += (-1 if a_val == last_B else 1)
    last_B = b_val

GPIO.setup(ENC_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENC_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# Using RISING edge for ENC_B might be simpler if your encoder gives clean pulses
# GPIO.add_event_detect(ENC_B, GPIO.RISING, callback=encoder_callback, bouncetime=5)
# Using BOTH for full quadrature, but ensure encoder_callback logic is correct for it.
# The original callback seems to try to implement quadrature direction detection.
# Simpler encoder: GPIO.add_event_detect(ENC_A, GPIO.RISING, callback=lambda x: global pulse_count; pulse_count += 1)

# For now, using the user's original encoder logic:
GPIO.add_event_detect(ENC_A, GPIO.BOTH, callback=encoder_callback)


# ─── CAMERA SETUP ─────────────────────────────────────────────
cap = None
for i in range(3): # Try camera indices 0, 1, 2
    cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cam.isOpened():
        cap = cam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        print(f"Camera opened on index {i}")
        break

if cap is None:
    print("No camera found on indices 0–2. Exiting.")
    GPIO.cleanup()
    sys.exit(1)

# ─── OBSTACLE DETECTION PARAMETERS ─────────────────────────────
OBSTACLE_MIN_AREA = 800  # Minimum contour area to be considered an obstacle
OBSTACLE_ROI_Y_START_FACTOR = 0.3 # Start detecting obstacles from 30% down the frame
OBSTACLE_ROI_Y_END_FACTOR = 0.9   # Up to 90% down the frame
OBSTACLE_ROI_X_CENTER_WIDTH_FACTOR = 0.5 # Central 50% of the width

# Durations for avoidance maneuvers
REVERSE_DURATION = 0.6
TURN_AVOID_DURATION = 0.7

# ─── OBSTACLE DETECTION FUNCTION ──────────────────────────────
def detect_obstacle_in_roi(frame_roi, min_area):
    """Detects obstacles in a given ROI of the frame."""
    gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    
    # Using Canny Edge Detection
    edges = cv2.Canny(blurred_roi, 50, 150)
    
    # Dilate to make edges thicker and join nearby edges
    kernel_dilate = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # For debugging, draw contours on the ROI (optional)
    # cv2.drawContours(frame_roi, contours, -1, (0,255,0), 1)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            # Optional: draw a bounding box around the detected obstacle
            # x, y, w, h_cnt = cv2.boundingRect(cnt)
            # cv2.rectangle(frame_roi, (x,y), (x+w_cnt, y+h_cnt), (0,0,255), 2)
            return True, dilated_edges # Return True and the processed image for display
    return False, dilated_edges

# ─── MAIN LOOP ────────────────────────────────────────────────
try:
    print("Starting robot. Press Ctrl+C to stop.")
    current_maneuver = None # To track if we are in an avoidance maneuver
    maneuver_end_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        h, w = frame.shape[:2]

        # --- Obstacle Detection ---
        # Define the central ROI for obstacle detection
        roi_y_start = int(h * OBSTACLE_ROI_Y_START_FACTOR)
        roi_y_end = int(h * OBSTACLE_ROI_Y_END_FACTOR)
        roi_x_center_half_width = int((w * OBSTACLE_ROI_X_CENTER_WIDTH_FACTOR) / 2)
        roi_x_start = w // 2 - roi_x_center_half_width
        roi_x_end = w // 2 + roi_x_center_half_width

        obstacle_roi_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # Draw the obstacle ROI on the main frame for visualization
        cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255,0,0), 2)

        obstacle_detected, processed_obstacle_roi = detect_obstacle_in_roi(obstacle_roi_frame.copy(), OBSTACLE_MIN_AREA) # Pass a copy

        # --- Decision Making ---
        if time.time() < maneuver_end_time:
            # Currently performing a maneuver, continue it
            if current_maneuver == "reversing":
                reverse(AVOID_SPEED)
            elif current_maneuver == "turning_left_avoid":
                turn_left(TURN_SPEED)
            elif current_maneuver == "turning_right_avoid":
                turn_right(TURN_SPEED)
            # No need to do anything else until maneuver time is up
        
        elif obstacle_detected:
            print("! OBSTACLE DETECTED !")
            stop()
            time.sleep(0.1) # Short pause

            # Start avoidance: reverse first
            print("  Reversing...")
            current_maneuver = "reversing"
            maneuver_end_time = time.time() + REVERSE_DURATION
            reverse(AVOID_SPEED) # Start reversing

            # After reversing, we'll decide to turn. Let's pre-plan the turn.
            # For simplicity, always turn left after reversing.
            # A more advanced way would be to check side ROIs.
            # This logic will be hit *after* reversing is done in the next loop iteration.
            # So we need to chain maneuvers.
            # Let's make it a sequence: reverse, then turn
        
        elif current_maneuver == "reversing" and time.time() >= maneuver_end_time:
            # Finished reversing, now turn
            stop()
            time.sleep(0.1)
            print("  Turning left to avoid...")
            current_maneuver = "turning_left_avoid" # Or choose randomly/based on side sensors
            maneuver_end_time = time.time() + TURN_AVOID_DURATION
            turn_left(TURN_SPEED)

        elif (current_maneuver == "turning_left_avoid" or current_maneuver == "turning_right_avoid") and time.time() >= maneuver_end_time:
            # Finished turning, clear maneuver
            stop()
            time.sleep(0.1)
            print("  Avoidance maneuver complete.")
            current_maneuver = None
            maneuver_end_time = 0
            forward(SPEED) # Try moving forward again
            time.sleep(0.2) # Move forward briefly before re-evaluating

        else: # No obstacle and not in a maneuver, proceed with line following (or default behavior)
            current_maneuver = None # Ensure state is cleared

            # --- Line Following Logic (from original code, adapted) ---
            # ROI for line following (bottom half of the image)
            line_roi_h_start = h // 2
            line_roi = frame[line_roi_h_start : h, :]
            
            gray_line = cv2.cvtColor(line_roi, cv2.COLOR_BGR2GRAY)
            # Threshold for dark line on light background
            _, mask_line = cv2.threshold(gray_line, 60, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            mask_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, kernel)
            mask_line = cv2.morphologyEx(mask_line, cv2.MORPH_CLOSE, kernel)

            cnts_line, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            line_roi_display = cv2.cvtColor(mask_line, cv2.COLOR_GRAY2BGR) # For display

            line_detected_this_frame = False
            if cnts_line:
                c = max(cnts_line, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] > 200: # Min area for line segment (was 1000, might be too large)
                    cx = int(M["m10"] / M["m00"])
                    cv2.circle(line_roi_display, (cx, int(M["m01"] / M["m00"])), 5, (0,0,255), -1) # Draw centroid
                    line_detected_this_frame = True

                    line_roi_w = line_roi.shape[1]
                    if cx < line_roi_w * 0.4:
                        print("Line: ← turn_left")
                        turn_left(SPEED) # Use default SPEED or a specific line_follow_turn_speed
                    elif cx > line_roi_w * 0.6:
                        print("Line: → turn_right")
                        turn_right(SPEED)
                    else:
                        print("Line: →→ forward")
                        forward(SPEED)
                else:
                    print("Line: Tiny blob, moving forward cautiously.")
                    forward(SPEED // 2) # Move slower if line is weak
            
            if not line_detected_this_frame:
                print("No line detected, moving forward.")
                forward(SPEED)

        # --- Display ---
        # Combine ROIs for display if you want
        # For obstacle ROI, convert processed_obstacle_roi to BGR if it's grayscale/binary
        if 'processed_obstacle_roi' in locals() and processed_obstacle_roi is not None:
            if len(processed_obstacle_roi.shape) == 2: # Grayscale
                 processed_obstacle_roi_bgr = cv2.cvtColor(processed_obstacle_roi, cv2.COLOR_GRAY2BGR)
            else: # Already BGR
                 processed_obstacle_roi_bgr = processed_obstacle_roi
            frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = processed_obstacle_roi_bgr


        # Display line following ROI if available
        if 'line_roi_display' in locals() and line_roi_display is not None:
             frame[line_roi_h_start : h, :] = cv2.addWeighted(frame[line_roi_h_start : h, :], 0.7, line_roi_display, 0.3, 0)


        cv2.imshow("Robot View", frame)
        # If you want to see the obstacle processing steps:
        # cv2.imshow("Obstacle Edges", processed_obstacle_roi) 
        # cv2.imshow("Line Mask", mask_line)


        # print(f"Encoder pulses: {pulse_count}") # Optional: print encoder
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
        
        time.sleep(0.01) # Loop delay

except KeyboardInterrupt:
    print("Program stopped by user (Ctrl+C)")

finally:
    print("Cleaning up GPIO and camera...")
    stop()
    GPIO.cleanup()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete. Bye!")
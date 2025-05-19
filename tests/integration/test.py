import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import sys

# ─── GPIO & MOTOR SETUP ───────────────────────────────────────
IN1, IN2 = 21, 20   # Left motor pins
IN3, IN4 = 19, 16   # Right motor pins
PWM_FREQ    = 500  # PWM frequency in Hz
SPEED        = 60  # Default forward speed (%)
TURN_SPEED   = 70  # Speed for turning maneuvers (%)
AVOID_SPEED  = 50  # Speed for avoidance maneuvers (%)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in (IN1, IN2, IN3, IN4):
    GPIO.setup(pin, GPIO.OUT)

# Initialize PWM outputs
p1 = GPIO.PWM(IN1, PWM_FREQ)
p2 = GPIO.PWM(IN2, PWM_FREQ)
p3 = GPIO.PWM(IN3, PWM_FREQ)
p4 = GPIO.PWM(IN4, PWM_FREQ)
for pwm in (p1, p2, p3, p4):
    pwm.start(0)

# ─── MOTOR CONTROL FUNCTIONS ─────────────────────────────────
def stop():
    """Stops all motors."""
    for pwm in (p1, p2, p3, p4):
        pwm.ChangeDutyCycle(0)

def forward(speed=SPEED):
    """Drives forward at given speed."""
    p1.ChangeDutyCycle(speed); p2.ChangeDutyCycle(0)
    p3.ChangeDutyCycle(speed); p4.ChangeDutyCycle(0)

def reverse(speed=SPEED):
    """Reverses at given speed."""
    p1.ChangeDutyCycle(0); p2.ChangeDutyCycle(speed)
    p3.ChangeDutyCycle(0); p4.ChangeDutyCycle(speed)

def turn_left(speed=TURN_SPEED):
    """Pivots left in place."""
    p1.ChangeDutyCycle(0); p2.ChangeDutyCycle(speed)
    p3.ChangeDutyCycle(speed); p4.ChangeDutyCycle(0)

def turn_right(speed=TURN_SPEED):
    """Pivots right in place."""
    p1.ChangeDutyCycle(speed); p2.ChangeDutyCycle(0)
    p3.ChangeDutyCycle(0); p4.ChangeDutyCycle(speed)

# ─── ENCODER SETUP ───────────────────────────────────────────
ENC_A, ENC_B = 14, 15
pulse_count = 0
GPIO.setup(ENC_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENC_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def encoder_callback(channel):
    """Updates pulse_count based on quadrature encoder inputs."""
    global pulse_count
    a = GPIO.input(ENC_A)
    b = GPIO.input(ENC_B)
    # Simple quadrature decoding
    if channel == ENC_A:
        pulse_count += 1 if a == b else -1
    else:
        pulse_count += 1 if a != b else -1

# Attach interrupts
GPIO.add_event_detect(ENC_A, GPIO.BOTH, callback=encoder_callback, bouncetime=5)
GPIO.add_event_detect(ENC_B, GPIO.BOTH, callback=encoder_callback, bouncetime=5)

# ─── CAMERA INITIALIZATION ───────────────────────────────────
cap = None
for idx in range(3):
    cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cam.isOpened():
        cap = cam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        print(f"Camera opened on index {idx}")
        break

if cap is None:
    print("Error: No camera found (indices 0-2)")
    GPIO.cleanup()
    sys.exit(1)

# ─── PARAMETERS: ROI & DURATIONS ─────────────────────────────
OBSTACLE_MIN_AREA = 800
ROI_Y_START_FACT  = 0.3  # Start at 30% from top
ROI_Y_END_FACT    = 0.9  # End at 90% from top
ROI_X_WIDTH_FACT  = 0.5  # Central 50% of width
REVERSE_DURATION  = 0.6  # seconds
TURN_DURATION     = 0.7  # seconds

# ─── DETECTION FUNCTIONS ─────────────────────────────────────
def detect_obstacle(frame):
    """Returns (detected: bool, bbox, processed_edges)."""
    h, w = frame.shape[:2]
    ys = int(h * ROI_Y_START_FACT)
    ye = int(h * ROI_Y_END_FACT)
    xw = int(w * ROI_X_WIDTH_FACT / 2)
    xs = w//2 - xw; xe = w//2 + xw

    roi = frame[ys:ye, xs:xe]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    dil = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) > OBSTACLE_MIN_AREA:
            return True, (xs, ys, xe, ye), dil
    return False, (xs, ys, xe, ye), dil


def detect_line(frame):
    """Returns (cx or None, roi_line, mask_line)."""
    h, w = frame.shape[:2]
    roi = frame[h//2:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None, roi, mask

    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None, roi, mask

    cx = int(M['m10']/M['m00'])
    return cx, roi, mask

# ─── MAIN CONTROL LOOP ────────────────────────────────────────
try:
    print("Starting robot control (Ctrl+C to stop)")
    state = None
    maneuver_end = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            break

        frame = cv2.flip(frame, -1)
        now = time.time()

        # Obstacle detection
        obs, bbox, obs_proc = detect_obstacle(frame)
        xs, ys, xe, ye = bbox
        cv2.rectangle(frame, (xs, ys), (xe, ye), (255, 0, 0), 2)

        # State machine for maneuvers
        if state and now < maneuver_end:
            # Continue current maneuver
            if state == 'reversing': reverse(AVOID_SPEED)
            elif state == 'turning':  turn_left(TURN_SPEED)

        elif obs:
            # Initiate avoidance
            print("[!] Obstacle detected: reversing")
            stop(); time.sleep(0.1)
            state = 'reversing'
            maneuver_end = now + REVERSE_DURATION
            reverse(AVOID_SPEED)

        elif state == 'reversing' and now >= maneuver_end:
            print("[!] Reversal complete: turning")
            stop(); time.sleep(0.1)
            state = 'turning'
            maneuver_end = now + TURN_DURATION
            turn_left(TURN_SPEED)

        elif state == 'turning' and now >= maneuver_end:
            print("[!] Turn complete: moving forward")
            stop()
            forward(SPEED)
            state = None
            time.sleep(0.2)
        # can be delted If no obstacle and not in a maneuver, check line-following
        elif state is None:
            # Check if we are on the line
            cx, roi_line, mask_line = detect_line(frame)
            if cx is None:
                # Lost line: go forward
                forward(SPEED)
            else:
                lw = roi_line.shape[1]
                if cx < 0.4 * lw:
                    turn_left(SPEED)
                elif cx > 0.6 * lw:
                    turn_right(SPEED)
                else:
                    forward(SPEED)

        else:
            # Line-following default behavior
            cx, roi_line, mask_line = detect_line(frame)
            if cx is None:
                # Lost line: go forward
                forward(SPEED)
            else:
                lw = roi_line.shape[1]
                if cx < 0.4 * lw:
                    turn_left(SPEED)
                elif cx > 0.6 * lw:
                    turn_right(SPEED)
                else:
                    forward(SPEED)

        # Overlay obstacle processing result
        proc_bgr = cv2.cvtColor(obs_proc, cv2.COLOR_GRAY2BGR)
        frame[ys:ye, xs:xe] = proc_bgr

        # Display final frame
        cv2.imshow("Robot View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    print("Cleaning up...")
    stop()
    GPIO.cleanup()
    if cap: cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete")

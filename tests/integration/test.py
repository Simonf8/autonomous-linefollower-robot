import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import sys
import logging
from typing import Optional, Tuple

# ─── CONFIGURATION CONSTANTS ─────────────────────────────────────────────
IN1, IN2 = 21, 20   # Left motor pins
IN3, IN4 = 19, 26   # Right motor pins
ENC_A, ENC_B = 14, 15
PWM_FREQ = 500  # PWM frequency in Hz
SPEED = 60      # Default forward speed (%)
TURN_SPEED = 70 # Speed for turning maneuvers (%)
AVOID_SPEED = 50 # Speed for avoidance maneuvers (%)
OBSTACLE_MIN_AREA = 800
ROI_Y_START_FACT = 0.3  # Start at 30% from top
ROI_Y_END_FACT = 0.9    # End at 90% from top
ROI_X_WIDTH_FACT = 0.5  # Central 50% of width
REVERSE_DURATION = 0.6  # seconds
TURN_DURATION = 0.7     # seconds
CAM_WIDTH = 320
CAM_HEIGHT = 240

# ─── LOGGING SETUP ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── MOTOR CONTROL CLASS ─────────────────────────────────────────────────
class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (IN1, IN2, IN3, IN4):
            GPIO.setup(pin, GPIO.OUT)
        self.p1 = GPIO.PWM(IN1, PWM_FREQ)
        self.p2 = GPIO.PWM(IN2, PWM_FREQ)
        self.p3 = GPIO.PWM(IN3, PWM_FREQ)
        self.p4 = GPIO.PWM(IN4, PWM_FREQ)
        for pwm in (self.p1, self.p2, self.p3, self.p4):
            pwm.start(0)

    def stop(self) -> None:
        """Stops all motors."""
        for pwm in (self.p1, self.p2, self.p3, self.p4):
            pwm.ChangeDutyCycle(0)

    def forward(self, speed: int = SPEED) -> None:
        self.p1.ChangeDutyCycle(speed); self.p2.ChangeDutyCycle(0)
        self.p3.ChangeDutyCycle(speed); self.p4.ChangeDutyCycle(0)

    def reverse(self, speed: int = SPEED) -> None:
        self.p1.ChangeDutyCycle(0); self.p2.ChangeDutyCycle(speed)
        self.p3.ChangeDutyCycle(0); self.p4.ChangeDutyCycle(speed)

    def turn_left(self, speed: int = TURN_SPEED) -> None:
        self.p1.ChangeDutyCycle(0); self.p2.ChangeDutyCycle(speed)
        self.p3.ChangeDutyCycle(speed); self.p4.ChangeDutyCycle(0)

    def turn_right(self, speed: int = TURN_SPEED) -> None:
        self.p1.ChangeDutyCycle(speed); self.p2.ChangeDutyCycle(0)
        self.p3.ChangeDutyCycle(0); self.p4.ChangeDutyCycle(speed)

    def cleanup(self) -> None:
        self.stop()
        GPIO.cleanup()

# ─── ENCODER CLASS ───────────────────────────────────────────────────────
class Encoder:
    def __init__(self):
        self.pulse_count = 0
        GPIO.setup(ENC_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(ENC_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(ENC_A, GPIO.BOTH, callback=self.encoder_callback, bouncetime=5)
        GPIO.add_event_detect(ENC_B, GPIO.BOTH, callback=self.encoder_callback, bouncetime=5)

    def encoder_callback(self, channel):
        a = GPIO.input(ENC_A)
        b = GPIO.input(ENC_B)
        if channel == ENC_A:
            self.pulse_count += 1 if a == b else -1
        else:
            self.pulse_count += 1 if a != b else -1

# ─── CAMERA CLASS ────────────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.cap = None
        for idx in range(3):
            cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cam.isOpened():
                self.cap = cam
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                logger.info(f"Camera opened on index {idx}")
                break
        if self.cap is None:
            logger.error("No camera found (indices 0-2)")
            GPIO.cleanup()
            sys.exit(1)

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()

# ─── DETECTION FUNCTIONS ────────────────────────────────────────────────
def detect_obstacle(frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int], np.ndarray]:
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

def detect_line(frame: np.ndarray) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
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

# ─── LINE FOLLOWING LOGIC ───────────────────────────────────────────────
def line_following_logic(frame: np.ndarray, motors: MotorController) -> None:
    cx, roi_line, mask_line = detect_line(frame)
    if cx is None:
        motors.forward(SPEED)
    else:
        lw = roi_line.shape[1]
        if cx < 0.4 * lw:
            motors.turn_left(SPEED)
        elif cx > 0.6 * lw:
            motors.turn_right(SPEED)
        else:
            motors.forward(SPEED)

# ─── MAIN CONTROL LOOP ──────────────────────────────────────────────────
def main():
    motors = MotorController()
    encoder = Encoder()
    camera = Camera()
    state = None
    maneuver_end = 0
    try:
        logger.info("Starting robot control (Ctrl+C to stop)")
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.error("Frame grab failed")
                break
            frame = cv2.flip(frame, -1)
            now = time.time()
            # Obstacle detection
            obs, bbox, obs_proc = detect_obstacle(frame)
            xs, ys, xe, ye = bbox
            cv2.rectangle(frame, (xs, ys), (xe, ye), (255, 0, 0), 2)
            # State machine for maneuvers
            if state and now < maneuver_end:
                if state == 'reversing':
                    motors.reverse(AVOID_SPEED)
                elif state == 'turning':
                    motors.turn_left(TURN_SPEED)
            elif obs:
                logger.warning("Obstacle detected: reversing")
                motors.stop(); time.sleep(0.1)
                state = 'reversing'
                maneuver_end = now + REVERSE_DURATION
                motors.reverse(AVOID_SPEED)
            elif state == 'reversing' and now >= maneuver_end:
                logger.info("Reversal complete: turning")
                motors.stop(); time.sleep(0.1)
                state = 'turning'
                maneuver_end = now + TURN_DURATION
                motors.turn_left(TURN_SPEED)
            elif state == 'turning' and now >= maneuver_end:
                logger.info("Turn complete: moving forward")
                motors.stop()
                motors.forward(SPEED)
                state = None
                time.sleep(0.2)
            elif state is None:
                line_following_logic(frame, motors)
            else:
                line_following_logic(frame, motors)
            # Overlay obstacle processing result
            proc_bgr = cv2.cvtColor(obs_proc, cv2.COLOR_GRAY2BGR)
            frame[ys:ye, xs:xe] = proc_bgr
            # Display final frame
            cv2.imshow("Robot View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        logger.info("Cleaning up...")
        motors.cleanup()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()

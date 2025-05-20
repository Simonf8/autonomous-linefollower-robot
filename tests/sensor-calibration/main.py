# main.py -- MicroPython code for ESP32 to control Webots robot via HIL
# Receives line sensor data from Webots, runs line-following state machine, sends back robot state.

from machine import Pin, UART
import time

# ESP32 UART Setup for USB (REPL) communication:
# Initially use UART0 (USB REPL) so we can print messages and wait for user input
uart = UART(0, baudrate=115200, tx=1, rx=3)  # UART0 on pins TX=1, RX=3 (USB connector)

# Setup a user button and an LED for feedback (adjust GPIO numbers to your board)
button_pin = Pin(4, Pin.IN, Pin.PULL_UP)       # Example: pushbutton on GPIO4 (pulled up, press -> LOW)
led_board = Pin(2, Pin.OUT)                    # Example: on-board LED on GPIO2 (ESP32 DevKit V1)

# Prompt user to press the button to proceed
print("Press the button on the ESP32 to start HIL communication.")
print("Then close the serial monitor/Thonny and start the Webots simulation.")
print("Press Ctrl+C to cancel if needed.")
# Blink the LED waiting for user confirmation
while button_pin.value() == 1:  # wait while button not pressed (assuming LOW when pressed)
    led_board.value(not led_board.value())  # toggle LED
    time.sleep(0.25)
# Button was pressed: stop blinking and proceed
led_board.value(0)  # turn off LED (or on, as desired to indicate start)

# Reconfigure UART to use UART1 on the same pins, for communication with Webots
uart = UART(1, baudrate=115200, tx=1, rx=3)  # Switch to UART1 on TX=1, RX=3:contentReference[oaicite:11]{index=11}
# Note: After this point, the REPL console is lost, and the ESP32 communicates only with Webots.

# Line-following state definitions (single-character codes for states)
STATE_FORWARD = 'F'
STATE_LEFT    = 'L'
STATE_RIGHT   = 'R'
STATE_SEARCH  = 'S'

current_state = STATE_SEARCH   # start in 'search' (or idle) state until sensors received
previous_state = current_state

# Sensor status variables (True = line detected, False = no line)
line_left = False
line_center = False
line_right = False

# Optionally, track last turn direction for recovery (None, 'L', or 'R')
last_turn = None

# Main loop: continuously read sensor data, update state, and send commands
while True:
    # 1. Check for incoming sensor data from Webots
    if uart.any():  # if bytes available in serial buffer:contentReference[oaicite:12]{index=12}
        msg_bytes = uart.read()         # read all available bytes
        if msg_bytes is None:
            continue  # no data read, skip
        try:
            msg_str = msg_bytes.decode('utf-8')
        except Exception as e:
            # decoding error (if non-UTF8 bytes), skip this loop
            continue

        # Sensor messages are expected to be small (3 chars + newline each). 
        # There might be multiple messages if reading was slow; take the last complete one.
        data_lines = msg_str.strip().splitlines()
        if len(data_lines) == 0:
            continue
        latest = data_lines[-1]  # last line of data received
        if len(latest) >= 3:
            # Parse the last 3 characters as sensor bits (left, center, right)
            line_left   = True if latest[0] == '1' else False
            line_center = True if latest[1] == '1' else False
            line_right  = True if latest[2] == '1' else False
        # (If the latest line is shorter than 3 chars, ignore it as incomplete)

    # 2. State Machine: decide robot state based on sensor inputs (outer line follow logic)
    new_state = current_state  # default to keep the same state unless changed

    if line_center:  
        # Line is visible ahead – go straight
        new_state = STATE_FORWARD
        last_turn = None  # reset last turn direction since we're on track forward
    elif not line_center:
        # No line ahead; need to turn if possible
        if line_left and line_right:
            # Line detected on both sides (T-intersection or crossroads with no forward path).
            # Choose one direction as "outer" line. (Here we choose left by default; adjust if needed.)
            new_state = STATE_LEFT
            last_turn = 'L'
        elif line_left:
            # Line is to the left side
            new_state = STATE_LEFT
            last_turn = 'L'
        elif line_right:
            # Line is to the right side
            new_state = STATE_RIGHT
            last_turn = 'R'
        else:
            # line_center is False and neither side sees the line -> line is lost
            # Enter search state: continue turning in last known direction (or default) to find the line
            if last_turn == 'L':
                new_state = STATE_LEFT   # continue turning left to search
            elif last_turn == 'R':
                new_state = STATE_RIGHT  # continue turning right to search
            else:
                # No last turn info (we were going straight and lost the line) – choose a default direction
                new_state = STATE_LEFT
            # Note: new_state remains left/right, but we treat it as a searching turn
            # We use STATE_SEARCH code 'S' only if we wanted a different wheel behavior (e.g., slower spin).

    # (Optional) If you want a distinct behavior when truly lost, you could set new_state = STATE_SEARCH 
    # here instead of reusing left/right, and handle 'S' differently in Webots (e.g., spin in place).
    # In this simple approach, we just reuse L or R to keep turning.

    # 3. Send the new state to Webots if it has changed
    if new_state != current_state:
        current_state = new_state
        # Transmit the state code to Webots, followed by newline
        try:
            uart.write(current_state + '\n')
        except Exception as e:
            # If there's an error in write (e.g., serial disconnected), just continue
            pass
    # If state is unchanged, we do not send anything, to minimize communication:contentReference[oaicite:13]{index=13}

    # Small delay to avoid saturating CPU (adjust loop frequency as needed)
    time.sleep(0.01)

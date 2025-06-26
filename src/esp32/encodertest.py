import time
from machine import Pin, PWM

class EncoderTester:
    def __init__(self):
        # Motor pin setup (same as your main code)
        self.motors = {
            'fl': {'p1': PWM(Pin(5), freq=50), 'p2': PWM(Pin(4), freq=50)},
            'fr': {'p1': PWM(Pin(6), freq=50), 'p2': PWM(Pin(7), freq=50)},
            'bl': {'p1': PWM(Pin(16), freq=50), 'p2': PWM(Pin(15), freq=50)},
            'br': {'p1': PWM(Pin(17), freq=50), 'p2': PWM(Pin(18), freq=50)}
        }
        
        # Encoder pin setup (same as your main code)
        self.encoders = {
            'fl': {'a': Pin(38, Pin.IN, Pin.PULL_UP), 'b': Pin(39, Pin.IN, Pin.PULL_UP)},
            'fr': {'a': Pin(2, Pin.IN, Pin.PULL_UP), 'b': Pin(42, Pin.IN, Pin.PULL_UP)},
            'bl': {'a': Pin(41, Pin.IN, Pin.PULL_UP), 'b': Pin(40, Pin.IN, Pin.PULL_UP)},
            'br': {'a': Pin(0, Pin.IN, Pin.PULL_UP), 'b': Pin(45, Pin.IN, Pin.PULL_UP)}
        }
        
        # Tick counters
        self.ticks = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}
        
        self.setup_encoders()
        self.stop_all()
        
        print("Encoder Tester initialized")
        print("Motor pins: FL(5,4), FR(6,7), BL(16,15), BR(17,18)")
        print("Encoder pins: FL(38,39), FR(2,42), BL(41,40), BR(0,45)")

    def setup_encoders(self):
        """Setup encoder interrupt handlers"""
        for wheel, pins in self.encoders.items():
            pins['a'].irq(trigger=Pin.IRQ_RISING, handler=self.create_isr(wheel, pins))

    def create_isr(self, wheel, pins):
        """Creates a unique ISR for each wheel"""
        def isr(pin):
            if pins['b'].value():
                self.ticks[wheel] += 1  # Forward
            else:
                self.ticks[wheel] -= 1  # Reverse
        return isr

    def reset_ticks(self, wheel=None):
        """Reset tick counter for specific wheel or all wheels"""
        if wheel:
            self.ticks[wheel] = 0
        else:
            for w in self.ticks:
                self.ticks[w] = 0

    def set_motor_speed(self, wheel, speed):
        """Set speed for a single motor. Speed is from -100 to 100."""
        speed = max(-100, min(100, speed))
        duty = int(abs(speed) * 10.23)  # Scale to 0-1023

        motor = self.motors[wheel]
        if speed > 0:
            motor['p1'].duty(duty)
            motor['p2'].duty(0)
        elif speed < 0:
            motor['p1'].duty(0)
            motor['p2'].duty(duty)
        else:
            motor['p1'].duty(0)
            motor['p2'].duty(0)

    def stop_all(self):
        """Stop all motors"""
        for wheel in self.motors:
            self.set_motor_speed(wheel, 0)

    def test_single_wheel(self, wheel, speed=50, duration=3):
        """Test a single wheel and its encoder"""
        print(f"\n=== Testing {wheel.upper()} wheel ===")
        print(f"Motor pins: {5 if wheel=='fl' else 6 if wheel=='fr' else 16 if wheel=='bl' else 17}, "
              f"{4 if wheel=='fl' else 7 if wheel=='fr' else 15 if wheel=='bl' else 18}")
        print(f"Encoder pins: A={38 if wheel=='fl' else 2 if wheel=='fr' else 41 if wheel=='bl' else 0}, "
              f"B={39 if wheel=='fl' else 42 if wheel=='fr' else 40 if wheel=='bl' else 45}")
        
        # Reset this wheel's counter
        self.reset_ticks(wheel)
        initial_ticks = self.ticks[wheel]
        
        print(f"Initial ticks: {initial_ticks}")
        print(f"Running motor at speed {speed} for {duration} seconds...")
        
        # Run motor forward
        self.set_motor_speed(wheel, speed)
        
        # Monitor for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            current_ticks = self.ticks[wheel]
            elapsed = time.time() - start_time
            print(f"Time: {elapsed:.1f}s, Ticks: {current_ticks}, Rate: {current_ticks/elapsed:.1f} ticks/sec")
            time.sleep(0.5)
        
        # Stop motor
        self.set_motor_speed(wheel, 0)
        final_ticks = self.ticks[wheel]
        
        print(f"Final ticks: {final_ticks}")
        print(f"Total ticks in {duration}s: {final_ticks - initial_ticks}")
        print(f"Average rate: {(final_ticks - initial_ticks)/duration:.1f} ticks/sec")
        
        # Test reverse direction
        print(f"\nTesting reverse direction...")
        self.reset_ticks(wheel)
        self.set_motor_speed(wheel, -speed)
        time.sleep(2)
        self.set_motor_speed(wheel, 0)
        reverse_ticks = self.ticks[wheel]
        print(f"Reverse ticks after 2s: {reverse_ticks}")
        
        # Analysis
        if abs(final_ticks - initial_ticks) > 10:
            print("✓ Encoder appears to be working")
            if reverse_ticks < -10:
                print("✓ Direction detection working correctly")
            else:
                print("⚠ Direction detection may need checking")
        else:
            print("✗ Encoder may not be working - very few ticks detected")
        
        time.sleep(1)

    def test_all_wheels_individual(self):
        """Test each wheel individually"""
        print("Starting individual wheel tests...")
        print("Each wheel will run for 3 seconds forward, then 2 seconds reverse")
        
        wheels = ['fl', 'fr', 'bl', 'br']
        for wheel in wheels:
            self.test_single_wheel(wheel)
            
            # Ask user to continue or skip
            try:
                response = input(f"\nPress Enter to continue to next wheel, or 'q' to quit: ")
                if response.lower() == 'q':
                    break
            except:
                # If input() isn't available (some MicroPython environments)
                time.sleep(2)
        
        self.stop_all()

    def continuous_monitor(self, wheel):
        """Continuously monitor a specific wheel's encoder"""
        print(f"\n=== Continuous monitoring of {wheel.upper()} wheel ===")
        print("Manually spin the wheel and watch the tick count")
        print("Press Ctrl+C to stop monitoring")
        
        self.reset_ticks(wheel)
        last_ticks = 0
        
        try:
            while True:
                current_ticks = self.ticks[wheel]
                if current_ticks != last_ticks:
                    direction = "CW" if current_ticks > last_ticks else "CCW"
                    print(f"Ticks: {current_ticks:6d} (Change: {current_ticks - last_ticks:+3d}, {direction})")
                    last_ticks = current_ticks
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped. Final count: {self.ticks[wheel]}")

    def show_all_encoders(self):
        """Show real-time readings from all encoders"""
        print("\n=== All Encoder Readings ===")
        print("Manually spin wheels to test. Press Ctrl+C to stop.")
        
        self.reset_ticks()
        
        try:
            while True:
                print(f"FL:{self.ticks['fl']:6d} | FR:{self.ticks['fr']:6d} | "
                      f"BL:{self.ticks['bl']:6d} | BR:{self.ticks['br']:6d}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped monitoring all encoders")

def main():
    tester = EncoderTester()
    
    while True:
        print("\n" + "="*50)
        print("ENCODER TEST MENU")
        print("="*50)
        print("1. Test all wheels individually (automated)")
        print("2. Test specific wheel")
        print("3. Continuous monitor specific wheel")
        print("4. Show all encoder readings")
        print("5. Exit")
        
        try:
            choice = input("Enter choice (1-5): ")
        except:
            # Fallback for environments without input()
            print("Running automated test...")
            choice = "1"
        
        if choice == "1":
            tester.test_all_wheels_individual()
        
        elif choice == "2":
            print("Available wheels: fl, fr, bl, br")
            try:
                wheel = input("Enter wheel to test: ").lower()
                if wheel in ['fl', 'fr', 'bl', 'br']:
                    tester.test_single_wheel(wheel)
                else:
                    print("Invalid wheel selection")
            except:
                # Test FL as default
                tester.test_single_wheel('fl')
        
        elif choice == "3":
            print("Available wheels: fl, fr, bl, br")
            try:
                wheel = input("Enter wheel to monitor: ").lower()
                if wheel in ['fl', 'fr', 'bl', 'br']:
                    tester.continuous_monitor(wheel)
                else:
                    print("Invalid wheel selection")
            except:
                tester.continuous_monitor('fl')
        
        elif choice == "4":
            tester.show_all_encoders()
        
        elif choice == "5":
            break
        
        else:
            print("Invalid choice")
    
    tester.stop_all()
    print("Encoder testing complete")

if __name__ == "__main__":
    main()
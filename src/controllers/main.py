#!/usr/bin/env python3

import threading
import logging
import sys
from config import *
from robot_controller import RobotController
from web_server import create_app

def main():
    """Main entry point for the autonomous line-following robot."""
    print("Initializing Robot System...")
    
    # Initialize the core robot controller
    robot = RobotController()
    
    # Initialize the web server with the robot instance
    app = create_app(robot)
    
    # Start the robot logic thread
    robot_thread = threading.Thread(target=robot.run, daemon=True)
    robot_thread.start()
    
    print(f"Robot logic started in thread: {robot_thread.name}")
    print(f"Starting Web Interface on port 5000...")
    
    try:
        # Run the Flask app on the main thread
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.stop()
        print("Robot system halted.")

if __name__ == "__main__":
    main()
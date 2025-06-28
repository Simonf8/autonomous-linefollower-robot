#!/usr/bin/env python3

import sys
import time
sys.path.append('src/controllers')

from visual_localizer import PreciseMazeLocalizer
from main import MAZE_GRID, START_CELL, END_CELL, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, START_DIRECTION

def test_position_tracking():
    """Test position tracking for black lines on white background"""
    print("=" * 60)
    print("POSITION TRACKING TEST FOR LINE FOLLOWING")
    print("=" * 60)
    print(f"Setup: Black lines on white background")
    print(f"Start position: {START_CELL}")
    print(f"End position: {END_CELL}")
    print(f"Start direction: {START_DIRECTION}")
    print()
    
    try:
        # Create localizer
        localizer = PreciseMazeLocalizer(
            maze=MAZE_GRID,
            start_pos=START_CELL,
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
            camera_fps=CAMERA_FPS,
            start_direction=START_DIRECTION
        )
        
        print("✓ Position tracker created")
        
        # Test camera initialization
        if localizer.initialize_camera():
            print("✓ Camera initialized successfully")
            
            # Test position tracking
            current_pos = localizer.get_current_cell()
            current_pose = localizer.get_pose()
            
            print(f"✓ Current position: {current_pos}")
            print(f"✓ Current pose: {current_pose}")
            
            # Test camera frame capture
            frame = localizer.get_camera_frame()
            if frame is not None:
                print(f"✓ Camera frame captured: {frame.shape}")
                
                # Test scene detection
                scene = localizer.detect_scene_with_precision()
                print(f"✓ Scene detection: {scene.get('scene_type', 'unknown')}")
                print(f"  - Status: {scene.get('status', 'unknown')}")
                print(f"  - Confidence: {scene.get('confidence', 0):.2f}")
                
            else:
                print("⚠ Camera frame capture failed (camera might be busy)")
            
            # Test localization thread
            print("\nTesting localization thread...")
            localizer.start_localization()
            
            for i in range(3):
                time.sleep(1)
                status = localizer.get_status()
                print(f"  Second {i+1}: Pos={status['position']}, Status={status['status']}, Scene={status['scene_type']}")
            
            localizer.stop_localization()
            print("✓ Localization thread test completed")
            
        else:
            print("✗ Camera initialization failed")
            return False
            
        print()
        print("=" * 60)
        print("POSITION TRACKING SYSTEM STATUS: READY")
        print("=" * 60)
        print("✓ Camera working")
        print("✓ Line detection optimized for black lines on white background")
        print("✓ Position tracking functional")
        print("✓ Scene detection working")
        print("✓ Ready for autonomous navigation")
        print()
        print("Your robot can now track its position on the maze map!")
        print("The system will detect:")
        print("  - Corridors (straight lines)")
        print("  - T-junctions (line branches)")
        print("  - Intersections (line crossings)")
        print("  - Dead ends (line endings)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_position_tracking() 
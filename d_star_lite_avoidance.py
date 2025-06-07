#!/usr/bin/env python3
"""
Simplified D* Lite Inspired Obstacle Avoidance for Line Following Robot
Adapted from full D* Lite for line-following applications
"""

import math
from collections import deque
import time

class LocalGridMapper:
    """
    Creates a local grid map around the robot for dynamic obstacle avoidance
    Simplified version of D* Lite concepts for line following
    """
    
    def __init__(self, grid_size=7, cell_size=0.1):
        """
        Initialize local grid mapper
        grid_size: Size of local grid (grid_size x grid_size)
        cell_size: Real-world size of each cell in meters
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.center = grid_size // 2
        
        # Grid: 0 = free, 1 = obstacle, 2 = line path
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Cost map for pathfinding
        self.cost_map = [[float('inf') for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Initialize center column as line path
        for row in range(grid_size):
            self.grid[row][self.center] = 2  # Line path
            self.cost_map[row][self.center] = 1  # Low cost for line
    
    def update_obstacles(self, detected_objects, robot_position, line_position):
        """
        Update grid with detected obstacles
        detected_objects: List of objects from YOLO
        robot_position: Current robot position 
        line_position: Current line position in frame
        """
        # Clear previous obstacles (keep line path)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == 1:  # Previous obstacle
                    self.grid[row][col] = 0
                    self.cost_map[row][col] = 5  # Higher cost for previously occupied areas
        
        # Add new obstacles
        for obj in detected_objects:
            grid_x, grid_y = self._world_to_grid(obj['position'], obj.get('distance', 1.0))
            if self._is_valid_cell(grid_x, grid_y):
                self.grid[grid_y][grid_x] = 1  # Obstacle
                self.cost_map[grid_y][grid_x] = float('inf')  # Infinite cost
                
                # Add buffer around obstacle
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        bx, by = grid_x + dx, grid_y + dy
                        if self._is_valid_cell(bx, by) and self.grid[by][bx] != 1:
                            if self.grid[by][bx] != 2:  # Don't block line path
                                self.cost_map[by][bx] = 10  # High cost buffer
    
    def _world_to_grid(self, relative_position, distance):
        """Convert world position to grid coordinates"""
        # Simplified conversion - assumes robot is at bottom center
        grid_x = int(self.center + relative_position * 2)  # Scale factor
        grid_y = int(self.grid_size - 1 - distance * 2)    # Distance ahead
        return grid_x, grid_y
    
    def _is_valid_cell(self, x, y):
        """Check if cell coordinates are valid"""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def find_avoidance_path(self):
        """
        Find path around obstacles back to line using simplified A*
        Returns: list of (direction, steps) commands
        """
        start = (self.center, self.grid_size - 1)  # Robot position (bottom center)
        goal = (self.center, 0)  # Goal (top center - continue on line)
        
        # Simple A* pathfinding
        open_set = [(self._heuristic(start, goal), 0, start, [])]
        closed_set = set()
        
        while open_set:
            open_set.sort()
            f_score, g_score, current, path = open_set.pop(0)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == goal or g_score > 10:  # Limit search depth
                return self._convert_path_to_commands(path)
            
            # Explore neighbors
            for dx, dy in [(0, -1), (1, 0), (-1, 0), (0, 1)]:  # Up, Right, Left, Down
                nx, ny = current[0] + dx, current[1] + dy
                
                if not self._is_valid_cell(nx, ny) or (nx, ny) in closed_set:
                    continue
                
                move_cost = self.cost_map[ny][nx]
                if move_cost == float('inf'):
                    continue
                
                new_g_score = g_score + move_cost
                new_path = path + [(dx, dy)]
                h_score = self._heuristic((nx, ny), goal)
                f_score = new_g_score + h_score
                
                open_set.append((f_score, new_g_score, (nx, ny), new_path))
        
        return []  # No path found
    
    def _heuristic(self, pos1, pos2):
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _convert_path_to_commands(self, path):
        """Convert path to robot movement commands"""
        if not path:
            return []
        
        commands = []
        current_direction = (0, -1)  # Initially facing forward
        
        for move in path:
            if move != current_direction:
                # Need to turn
                if move == (1, 0):  # Right
                    commands.append('turn_right')
                elif move == (-1, 0):  # Left
                    commands.append('turn_left')
                elif move == (0, 1):  # Backward (shouldn't happen)
                    commands.append('turn_around')
                current_direction = move
            
            commands.append('forward')
        
        return commands


class DStarLiteAvoidance:
    """
    D* Lite inspired obstacle avoidance for line following robot
    Provides intelligent path planning around obstacles
    """
    
    def __init__(self):
        self.local_mapper = LocalGridMapper()
        self.planned_commands = deque()
        self.avoidance_active = False
        self.last_replan_time = 0
        self.replan_interval = 0.5  # Replan every 0.5 seconds
        
        # State tracking
        self.obstacle_memory = deque(maxlen=10)  # Remember obstacles
        self.clear_path_counter = 0
        
    def update_and_plan(self, detected_objects, line_detected, line_offset):
        """
        Main update function - processes obstacles and plans avoidance
        """
        current_time = time.time()
        
        # Update obstacle memory
        if detected_objects:
            self.obstacle_memory.extend(detected_objects)
            self.clear_path_counter = 0
        else:
            self.clear_path_counter += 1
        
        # Check if we need to replan
        need_replan = (
            (current_time - self.last_replan_time > self.replan_interval) or
            (detected_objects and not self.avoidance_active) or
            (not detected_objects and self.avoidance_active and self.clear_path_counter > 5)
        )
        
        if need_replan:
            self._replan(detected_objects, line_detected, line_offset)
            self.last_replan_time = current_time
        
        return self._get_next_command()
    
    def _replan(self, detected_objects, line_detected, line_offset):
        """Replan path around obstacles"""
        if detected_objects:
            # Obstacles detected - plan avoidance
            self.local_mapper.update_obstacles(detected_objects, 
                                             robot_position=(0, 0), 
                                             line_position=line_offset)
            
            new_commands = self.local_mapper.find_avoidance_path()
            
            if new_commands:
                self.planned_commands = deque(new_commands)
                self.avoidance_active = True
                print(f"🧭 D* Lite: Planning avoidance - {len(new_commands)} commands")
            else:
                # Fallback to simple avoidance
                self._plan_simple_avoidance(detected_objects[0])
                
        elif self.avoidance_active and self.clear_path_counter > 5:
            # Path is clear - return to line following
            if line_detected:
                return_commands = self._plan_return_to_line(line_offset)
                self.planned_commands = deque(return_commands)
                print(f"🧭 D* Lite: Returning to line - offset: {line_offset:.2f}")
            else:
                self.avoidance_active = False
                self.planned_commands.clear()
                print(f"🧭 D* Lite: Avoidance complete")
    
    def _plan_simple_avoidance(self, main_obstacle):
        """Fallback simple avoidance when pathfinding fails"""
        commands = []
        
        if main_obstacle['position'] < -0.1:  # Obstacle on left
            commands = ['turn_right', 'forward', 'forward', 'turn_left', 'forward', 'turn_left']
        elif main_obstacle['position'] > 0.1:  # Obstacle on right
            commands = ['turn_left', 'forward', 'forward', 'turn_right', 'forward', 'turn_right']
        else:  # Obstacle in center
            commands = ['turn_right', 'forward', 'forward', 'turn_left', 'forward', 'forward', 'turn_left']
        
        self.planned_commands = deque(commands)
        self.avoidance_active = True
        print(f"🧭 D* Lite: Simple avoidance fallback")
    
    def _plan_return_to_line(self, line_offset):
        """Plan return to line based on current offset"""
        commands = []
        
        if abs(line_offset) > 0.3:  # Far from line
            if line_offset > 0:  # Line is to the right
                commands = ['turn_right', 'forward']
            else:  # Line is to the left
                commands = ['turn_left', 'forward']
        
        return commands
    
    def _get_next_command(self):
        """Get next movement command from planned sequence"""
        if self.planned_commands:
            command = self.planned_commands.popleft()
            
            if not self.planned_commands:  # Last command executed
                self.avoidance_active = False
                print(f"🧭 D* Lite: Command sequence complete")
            
            return self._convert_to_robot_command(command)
        
        return None  # No active avoidance
    
    def _convert_to_robot_command(self, d_star_command):
        """Convert D* Lite command to robot movement command"""
        command_map = {
            'forward': 'FORWARD',
            'turn_left': 'AVOID_LEFT', 
            'turn_right': 'AVOID_RIGHT',
            'turn_around': 'AVOID_RIGHT'  # Turn around = multiple rights
        }
        
        return command_map.get(d_star_command, 'FORWARD')
    
    def is_avoidance_active(self):
        """Check if D* Lite avoidance is currently active"""
        return self.avoidance_active
    
    def get_status(self):
        """Get current status for debugging"""
        return {
            'avoidance_active': self.avoidance_active,
            'commands_remaining': len(self.planned_commands),
            'next_command': list(self.planned_commands)[0] if self.planned_commands else None,
            'obstacles_in_memory': len(self.obstacle_memory),
            'clear_path_count': self.clear_path_counter
        }


# Integration example for main.py
def integrate_d_star_lite(detected_objects, line_detected, line_offset, steering):
    """
    Integration function to use D* Lite avoidance in the main robot code
    
    Usage in main.py:
    # Initialize once
    d_star_avoidance = DStarLiteAvoidance()
    
    # In main loop
    d_star_command = integrate_d_star_lite(detected_objects, line_detected, line_offset, steering)
    if d_star_command:
        turn_command = d_star_command  # Use D* Lite command
    else:
        turn_command = normal_line_following_command  # Use normal PID
    """
    
    # This would be a global instance in the real implementation
    if not hasattr(integrate_d_star_lite, 'avoidance'):
        integrate_d_star_lite.avoidance = DStarLiteAvoidance()
    
    avoidance = integrate_d_star_lite.avoidance
    
    # Get D* Lite command
    d_star_command = avoidance.update_and_plan(detected_objects, line_detected, line_offset)
    
    return d_star_command


if __name__ == "__main__":
    # Test the D* Lite avoidance system
    avoidance = DStarLiteAvoidance()
    
    # Simulate obstacle detection
    test_objects = [{'position': 0.2, 'distance': 1.0, 'class_name': 'bottle'}]
    
    for i in range(10):
        if i < 3:
            # Obstacles present
            command = avoidance.update_and_plan(test_objects, True, 0.1)
        else:
            # Obstacles cleared
            command = avoidance.update_and_plan([], True, 0.2)
        
        status = avoidance.get_status()
        print(f"Step {i}: Command={command}, Status={status}")
        time.sleep(0.1) 
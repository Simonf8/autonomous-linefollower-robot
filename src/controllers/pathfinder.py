#!/usr/bin/env python3

import heapq
import logging
from typing import List, Tuple, Optional, Set, Dict

class Pathfinder:
    """
    Dijkstra-based pathfinding for grid-based navigation with dynamic replanning.
    """
    
    def __init__(self, grid: List[List[int]], cell_size_m: float = 0.025, turn_penalty: float = 3.0):
        """
        Initialize pathfinder.
        
        Args:
            grid: 2D grid where 0 = path, 1 = obstacle
            cell_size_m: Width of each grid cell in meters
            turn_penalty: Cost penalty for each turn (higher = prefer straight lines more)
        """
        self.original_grid = [row[:] for row in grid]  
        self.grid = [row[:] for row in grid]
        self.cell_size_m = cell_size_m
        
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        
        # Turn penalty - increase this to prefer straighter paths
        # 0 = shortest path, 1-2 = slight preference, 3-5 = strong preference, >5 = very strong preference
        self.turn_penalty = turn_penalty
        
        # Cache for computed distances
        self._distance_cache = {}
        self._last_goal = None
    
    @staticmethod
    def create_maze_grid() -> List[List[int]]:
        """Creates the default maze grid layout."""
     
        maze = [
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 0
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 1
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 2
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 3
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 4
            [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], # Row 5
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 6
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 7
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 8
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0], # Row 9
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 10
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 11
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 12
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 13
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0]  # Row 14
        ]
        # Return the maze as-is, no horizontal flip needed
        return maze
    
    def get_grid(self) -> List[List[int]]:
        """Return the current grid."""
        return self.grid
    
    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell coordinates are valid and passable."""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.grid[y][x] == 0)
        
                
    
    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighboring cells (adjacent cells only for proper corridor following).
        Returns a list of tuples, where each tuple contains the neighbor cell and the cost to move to it.
        """
        x, y = cell
        neighbors = []
        
        # Check only adjacent cells (1 unit away) to ensure proper corridor following
        # Format: (dx, dy, cost)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            
            if self.is_valid_cell(nx, ny):
                # Base cost is 1, but we'll modify this in find_path_prefer_straight
                neighbors.append(((nx, ny), 1))
        
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], prefer_straight: bool = True) -> Optional[List[Tuple[int, int]]]:
        """
        Find path using modified Dijkstra's algorithm.
        
        Args:
            start: Starting cell (x, y)
            goal: Goal cell (x, y)
            prefer_straight: If True, prefers paths with fewer turns
            
        Returns:
            List of cells from start to goal, or None if no path exists
        """
        if prefer_straight:
            return self.find_path_prefer_straight(start, goal)
        else:
            return self._find_path_original(start, goal)
    
    def _find_path_original(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Original shortest path using Dijkstra's algorithm.
        """
        if not self.is_valid_cell(start[0], start[1]):
            logging.error(f"Pathfinder Error: Start cell {start} is not valid. It's either out of bounds or on an obstacle.")
            return None
        
        if not self.is_valid_cell(goal[0], goal[1]):
            logging.error(f"Pathfinder Error: Goal cell {goal} is not valid. It's either out of bounds or on an obstacle.")
            return None
        
        if start == goal:
            return [start]
        
        # Priority queue: (distance, cell)
        pq = [(0, start)]
        distances = {start: 0}
        came_from = {}
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]  # Reverse to get start->goal
            
            # Check all neighbors, now with variable costs
            for neighbor, cost in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                new_dist = current_dist + cost
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current
                    # Use A* style priority with heuristic for better performance
                    priority = new_dist + self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (priority, neighbor))
        
        # No path found
        logging.warning(f"No path found from {start} to {goal}")
        return None
    
    def find_path_prefer_straight(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find path that prefers long straight lines and avoids sharp corners.
        Uses modified A* with turn penalty.
        """
        if not self.is_valid_cell(start[0], start[1]) or not self.is_valid_cell(goal[0], goal[1]):
            return None
        
        if start == goal:
            return [start]
        
        # Priority queue: (cost, turn_count, cell, direction_from_parent)
        pq = [(0, 0, start, None)]
        
        # Store the best cost and turn count to reach each cell
        best_cost = {start: (0, 0)}  # (cost, turn_count)
        came_from = {}
        came_from_direction = {}  # Track the direction we came from
        
        # Use the configured turn penalty
        TURN_PENALTY = self.turn_penalty
        
        while pq:
            current_cost, turn_count, current, from_direction = heapq.heappop(pq)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]
            
            # Skip if we've found a better path to this cell
            if current in best_cost:
                stored_cost, stored_turns = best_cost[current]
                # Allow slightly higher cost if it has fewer turns
                if current_cost > stored_cost + TURN_PENALTY and turn_count >= stored_turns:
                    continue
            
            # Check all neighbors
            for neighbor, base_cost in self.get_neighbors(current):
                # Calculate direction to neighbor
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                
                # Determine direction (N, S, E, W)
                if dx == 1:
                    to_direction = 'E'
                elif dx == -1:
                    to_direction = 'W'
                elif dy == 1:
                    to_direction = 'S'
                elif dy == -1:
                    to_direction = 'N'
                else:
                    continue
                
                # Check if this is a turn
                is_turn = from_direction is not None and from_direction != to_direction
                new_turn_count = turn_count + (1 if is_turn else 0)
                
                # Calculate cost with turn penalty
                move_cost = base_cost
                if is_turn:
                    move_cost += TURN_PENALTY
                
                new_cost = current_cost + move_cost
                
                # Check if this is a better path to the neighbor
                if neighbor in best_cost:
                    stored_cost, stored_turns = best_cost[neighbor]
                    # Skip if this path is worse (more cost AND more or equal turns)
                    if new_cost >= stored_cost and new_turn_count >= stored_turns:
                        continue
                    # Also skip if cost is much higher even with fewer turns
                    if new_cost > stored_cost + TURN_PENALTY * 2:
                        continue
                
                # This is a better or comparable path
                best_cost[neighbor] = (new_cost, new_turn_count)
                came_from[neighbor] = current
                came_from_direction[neighbor] = to_direction
                
                # Priority includes both cost and heuristic
                priority = new_cost + self.heuristic(neighbor, goal)
                heapq.heappush(pq, (new_cost, new_turn_count, neighbor, to_direction))
        
        # No path found
        return None
    
    def world_to_cell(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell coordinates."""
        cell_x = int(world_x / self.cell_size_m)
        cell_y = int(world_y / self.cell_size_m)
        return (cell_x, cell_y)
    
    def cell_to_world(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        """Convert grid cell coordinates to world coordinates (center of cell)."""
        world_x = (cell_x + 0.5) * self.cell_size_m
        world_y = (cell_y + 0.5) * self.cell_size_m
        return (world_x, world_y)
    
    def update_obstacle(self, cell_x: int, cell_y: int, is_obstacle: bool):
        """
        Update grid with new obstacle information.
        
        Args:
            cell_x: X coordinate of cell
            cell_y: Y coordinate of cell
            is_obstacle: True to mark as obstacle, False to clear
        """
        if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
            self.grid[cell_y][cell_x] = 1 if is_obstacle else 0
            # Clear cache since grid changed
            self._distance_cache.clear()
    
    def clear_obstacles(self):
        """Reset grid to original state (remove all dynamic obstacles)."""
        self.grid = [row[:] for row in self.original_grid]
        self._distance_cache.clear()
    
    def get_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate path length in meters."""
        if not path or len(path) < 2:
            return 0.0
        
        return (len(path) - 1) * self.cell_size_m
    
    def is_path_blocked(self, path: List[Tuple[int, int]]) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Check if a path is blocked by obstacles.
        
        Args:
            path: List of cells in the path
            
        Returns:
            Tuple of (is_blocked, first_blocked_cell)
        """
        for cell in path:
            if not self.is_valid_cell(cell[0], cell[1]):
                return True, cell
        return False, None
    
    def find_alternative_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                            blocked_cells: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        Find alternative path avoiding specific blocked cells.
        
        Args:
            start: Starting cell
            goal: Goal cell  
            blocked_cells: Set of cells to avoid
            
        Returns:
            Alternative path or None
        """
        # Temporarily mark blocked cells as obstacles
        original_values = {}
        for cell in blocked_cells:
            if 0 <= cell[0] < self.width and 0 <= cell[1] < self.height:
                original_values[cell] = self.grid[cell[1]][cell[0]]
                self.grid[cell[1]][cell[0]] = 1
        
        # Find path with temporary obstacles
        path = self.find_path(start, goal)
        
        # Restore original values
        for cell, value in original_values.items():
            self.grid[cell[1]][cell[0]] = value
        
        return path
    
    def get_grid_info(self) -> Dict:
        """Get information about the current grid state."""
        total_cells = self.width * self.height
        obstacle_cells = sum(sum(row) for row in self.grid)
        free_cells = total_cells - obstacle_cells
        
        return {
            'width': self.width,
            'height': self.height,
            'total_cells': total_cells,
            'obstacle_cells': obstacle_cells,
            'free_cells': free_cells,
            'obstacle_percentage': (obstacle_cells / total_cells) * 100
        }
    
    def count_turns_in_path(self, path: List[Tuple[int, int]]) -> int:
        """Count the number of turns in a given path."""
        if len(path) < 3:
            return 0
        
        turns = 0
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_cell = path[i + 1]
            
            # Calculate direction vectors
            dir1 = (curr[0] - prev[0], curr[1] - prev[1])
            dir2 = (next_cell[0] - curr[0], next_cell[1] - curr[1])
            
            # If directions are different, it's a turn
            if dir1 != dir2:
                turns += 1
        
        return turns
    
    def get_path_segments(self, path: List[Tuple[int, int]]) -> List[Tuple[int, str]]:
        """
        Break down a path into straight segments.
        Returns list of (length, direction) tuples.
        """
        if len(path) < 2:
            return []
        
        segments = []
        current_direction = None
        segment_length = 0
        
        for i in range(len(path) - 1):
            curr = path[i]
            next_cell = path[i + 1]
            
            # Calculate direction
            dx = next_cell[0] - curr[0]
            dy = next_cell[1] - curr[1]
            
            if dx == 1:
                direction = 'E'
            elif dx == -1:
                direction = 'W'
            elif dy == 1:
                direction = 'S'
            elif dy == -1:
                direction = 'N'
            else:
                continue
            
            if direction == current_direction:
                segment_length += 1
            else:
                if current_direction is not None:
                    segments.append((segment_length, current_direction))
                current_direction = direction
                segment_length = 1
        
        # Add the last segment
        if current_direction is not None:
            segments.append((segment_length, current_direction))
        
        return segments 

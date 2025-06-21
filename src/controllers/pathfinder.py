#!/usr/bin/env python3

import heapq
import logging
from typing import List, Tuple, Optional, Set, Dict

class Pathfinder:
    """
    Dijkstra-based pathfinding for grid-based navigation with dynamic replanning.
    """
    
    def __init__(self, grid: List[List[int]], cell_size_m: float = 0.025):
        """
        Initialize pathfinder.
        
        Args:
            grid: 2D grid where 0 = path, 1 = obstacle
            cell_size_m: Width of each grid cell in meters
        """
        self.original_grid = [row[:] for row in grid]  # Keep original for reset
        self.grid = [row[:] for row in grid]  # Working copy
        self.cell_size_m = cell_size_m
        
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        
        # Cache for computed distances
        self._distance_cache = {}
        self._last_goal = None
    
    @staticmethod
    def create_maze_grid() -> List[List[int]]:
        """Creates the default maze grid layout."""
        # The maze is defined with 0=path, 1=obstacle.
        # Some paths were defined with alternating 0s and 1s (e.g., [0,1,0,1,...])
        # which is not navigable by an algorithm that checks immediate neighbors.
        # The paths have been changed to be solid (all 0s).
        maze = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 0 - Solid path
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 1 - Solid path
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 2
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 3
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 4
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0], # Row 5
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 6
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 7
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 8
            [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], # Row 9
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 10
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 11
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 12
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 13 - Solid path
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]  # Row 14 - Solid path
        ]
        # Flip horizontally to match coordinate system
        return [row[::-1] for row in maze]
    
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
        Get valid neighboring cells, handling both solid and dashed lines.
        Returns a list of tuples, where each tuple contains the neighbor cell and the cost to move to it.
        """
        x, y = cell
        neighbors = []
        
        # Check for neighbors 1 unit away (for solid paths) and 2 units away (for dashed paths).
        # Format: (dx, dy, cost)
        for dx, dy, cost in [(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1),
                             (0, -2, 2), (0, 2, 2), (-2, 0, 2), (2, 0, 2)]:
            nx, ny = x + dx, y + dy
            
            # For a 2-step move, ensure the intermediate cell is not a path.
            # This prevents jumping over valid intersections on solid paths.
            if cost == 2:
                ix, iy = x + dx // 2, y + dy // 2
                if self.is_valid_cell(ix, iy):
                    continue # Don't jump over a valid path cell

            if self.is_valid_cell(nx, ny):
                # Avoid adding duplicates. If a 1-step neighbor was already found,
                # a 2-step check in the same direction is redundant or invalid.
                is_duplicate = False
                for neighbor_cell, _ in neighbors:
                    if neighbor_cell == (nx, ny):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path using Dijkstra's algorithm.
        
        Args:
            start: Starting cell (x, y)
            goal: Goal cell (x, y)
            
        Returns:
            List of cells from start to goal, or None if no path exists
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
#!/usr/bin/env python3

import time
import logging
from typing import List, Tuple, Optional, Dict
from enum import Enum

class BoxState(Enum):
    """States for box collection and delivery."""
    AVAILABLE = "available"
    COLLECTED = "collected"
    DELIVERED = "delivered"

class BoxHandler:
    """Manages box pickup and delivery operations."""
    
    def __init__(self, pickup_locations: List[Tuple[int, int]], 
                 dropoff_locations: List[Tuple[int, int]]):
        """
        Initialize box handler.
        
        Args:
            pickup_locations: List of pickup cell coordinates
            dropoff_locations: List of dropoff cell coordinates
        """
        self.pickup_locations = pickup_locations.copy()
        self.dropoff_locations = dropoff_locations.copy()
        
        # Track box states
        self.box_states = {}
        for i, pickup in enumerate(pickup_locations):
            self.box_states[f'P{i+1}'] = {
                'pickup_location': pickup,
                'dropoff_location': dropoff_locations[i] if i < len(dropoff_locations) else None,
                'state': BoxState.AVAILABLE,
                'collected_time': None,
                'delivered_time': None
            }
        
        # Mission state
        self.current_pickup_index = 0
        self.current_dropoff_index = 0
        self.has_package = False
        self.current_box_id = None
        
        # Statistics
        self.total_pickup_time = 0.0
        self.total_delivery_time = 0.0
        self.mission_start_time = None
    
    def start_mission(self, silent=False):
        """Start the box collection mission."""
        self.mission_start_time = time.time()
        if not silent:
            self._print_mission_overview()
    
    def _print_mission_overview(self):
        """Print mission overview."""
        print("=" * 60)
        print("MULTI-BOX DELIVERY MISSION STARTED")
        print(f"Total boxes to collect: {len(self.pickup_locations)}")
        print(f"Pickup locations: {[f'P{i+1}' for i in range(len(self.pickup_locations))]}")
        print(f"Dropoff locations: {[f'D{i+1}' for i in range(len(self.dropoff_locations))]}")
        print("=" * 60)
    
    def get_current_target(self) -> Optional[Tuple[Tuple[int, int], str]]:
        """
        Get the current target location and mission type.
        
        Returns:
            Tuple of (target_cell, mission_type) or None if mission complete
        """
        if self.has_package:
            # Need to deliver current package
            if self.current_dropoff_index < len(self.dropoff_locations):
                target = self.dropoff_locations[self.current_dropoff_index]
                return (target, "DROPOFF")
        else:
            # Need to collect next package
            if self.current_pickup_index < len(self.pickup_locations):
                target = self.pickup_locations[self.current_pickup_index]
                return (target, "PICKUP")
        
        return None  # Mission complete
    
    def collect_package(self, pickup_location: Tuple[int, int], simulation_delay: float = 1.0) -> bool:
        """
        Simulate collecting a package at pickup location.
        
        Args:
            pickup_location: Cell coordinates of pickup
            simulation_delay: Time to simulate pickup operation
            
        Returns:
            True if collection successful
        """
        if self.has_package:
            return False
        
        # Find which box this corresponds to
        box_id = None
        for bid, box_info in self.box_states.items():
            if (box_info['pickup_location'] == pickup_location and 
                box_info['state'] == BoxState.AVAILABLE):
                box_id = bid
                break
        
        if not box_id:
            return False
        
        # Simulate pickup operation
        print(f"Collecting package {box_id} at location {pickup_location}...")
        time.sleep(simulation_delay)
        
        # Update state
        self.box_states[box_id]['state'] = BoxState.COLLECTED
        self.box_states[box_id]['collected_time'] = time.time()
        self.has_package = True
        self.current_box_id = box_id
        self.current_pickup_index += 1
        
        progress = self.get_collection_progress()
        print(f"Package {box_id} collected! Progress: {progress['collected']}/{progress['total']}")
        
        return True
    
    def deliver_package(self, dropoff_location: Tuple[int, int], simulation_delay: float = 1.0) -> bool:
        """
        Simulate delivering a package at dropoff location.
        
        Args:
            dropoff_location: Cell coordinates of dropoff
            simulation_delay: Time to simulate delivery operation
            
        Returns:
            True if delivery successful
        """
        if not self.has_package or not self.current_box_id:
            return False
        
        # Verify this is the correct dropoff location
        box_info = self.box_states[self.current_box_id]
        if box_info['dropoff_location'] != dropoff_location:
            return False
        
        # Simulate delivery operation
        print(f"Delivering package {self.current_box_id} to location {dropoff_location}...")
        time.sleep(simulation_delay)
        
        # Update state
        box_info['state'] = BoxState.DELIVERED
        box_info['delivered_time'] = time.time()
        self.has_package = False
        self.current_dropoff_index += 1
        
        # Calculate delivery time
        if box_info['collected_time']:
            delivery_time = box_info['delivered_time'] - box_info['collected_time']
            self.total_delivery_time += delivery_time
            print(f"Package {self.current_box_id} delivered in {delivery_time:.1f}s")
        
        self.current_box_id = None
        
        progress = self.get_delivery_progress()
        print(f"Delivery complete! Progress: {progress['delivered']}/{progress['total']}")
        
        return True
    
    def is_at_pickup_location(self, current_cell: Tuple[int, int], tolerance: int = 1) -> bool:
        """
        Check if robot is at any pickup location.
        
        Args:
            current_cell: Robot's current cell coordinates
            tolerance: Distance tolerance in cells
            
        Returns:
            True if at a pickup location
        """
        for pickup_cell in self.pickup_locations:
            if (abs(current_cell[0] - pickup_cell[0]) <= tolerance and 
                abs(current_cell[1] - pickup_cell[1]) <= tolerance):
                return True
        return False
    
    def is_at_dropoff_location(self, current_cell: Tuple[int, int], tolerance: int = 1) -> bool:
        """
        Check if robot is at any dropoff location.
        
        Args:
            current_cell: Robot's current cell coordinates
            tolerance: Distance tolerance in cells
            
        Returns:
            True if at a dropoff location
        """
        for dropoff_cell in self.dropoff_locations:
            if (abs(current_cell[0] - dropoff_cell[0]) <= tolerance and 
                abs(current_cell[1] - dropoff_cell[1]) <= tolerance):
                return True
        return False
    
    def get_collection_progress(self) -> Dict:
        """Get current collection progress."""
        collected = sum(1 for box in self.box_states.values() 
                       if box['state'] in [BoxState.COLLECTED, BoxState.DELIVERED])
        return {
            'collected': collected,
            'total': len(self.box_states),
            'percentage': (collected / len(self.box_states)) * 100
        }
    
    def get_delivery_progress(self) -> Dict:
        """Get current delivery progress."""
        delivered = sum(1 for box in self.box_states.values() 
                       if box['state'] == BoxState.DELIVERED)
        return {
            'delivered': delivered,
            'total': len(self.box_states),
            'percentage': (delivered / len(self.box_states)) * 100
        }
    
    def is_mission_complete(self) -> bool:
        """Check if all boxes have been collected and delivered."""
        return all(box['state'] == BoxState.DELIVERED 
                  for box in self.box_states.values())
    
    def get_mission_statistics(self) -> Dict:
        """Get comprehensive mission statistics."""
        if not self.mission_start_time:
            return {}
        
        total_time = time.time() - self.mission_start_time
        collection_progress = self.get_collection_progress()
        delivery_progress = self.get_delivery_progress()
        
        # Calculate average times
        collected_boxes = [box for box in self.box_states.values() 
                          if box['collected_time']]
        delivered_boxes = [box for box in self.box_states.values() 
                          if box['delivered_time']]
        
        avg_pickup_time = 0
        if collected_boxes:
            pickup_times = [(box['collected_time'] - self.mission_start_time) 
                           for box in collected_boxes]
            avg_pickup_time = sum(pickup_times) / len(pickup_times)
        
        avg_delivery_time = 0
        if delivered_boxes:
            delivery_times = [(box['delivered_time'] - box['collected_time']) 
                           for box in delivered_boxes]
            avg_delivery_time = sum(delivery_times) / len(delivery_times)
            
        return {
            "total_time": total_time,
            "collection_progress": collection_progress,
            "delivery_progress": delivery_progress,
            "avg_pickup_time": avg_pickup_time,
            "avg_delivery_time": avg_delivery_time
        }
    
    def print_mission_summary(self, silent=False):
        """Print final mission summary."""
        if silent:
            return
            
        stats = self.get_mission_statistics()
        
        if not stats:
            print("Mission has not started.")
            return
            
        print("\n" + "=" * 60)
        print("MISSION SUMMARY")
        print("=" * 60)
        
        print(f"Total Mission Time: {stats['total_time']:.1f}s")
        
        # Collection stats
        coll_prog = stats['collection_progress']
        print(f"Boxes Collected: {coll_prog['collected']}/{coll_prog['total']} "
              f"({coll_prog['percentage']:.1f}%)")
        
        # Delivery stats
        del_prog = stats['delivery_progress']
        print(f"Boxes Delivered: {del_prog['delivered']}/{del_prog['total']} "
              f"({del_prog['percentage']:.1f}%)")
        
        # Performance
        if stats['avg_pickup_time'] > 0:
            print(f"Average Pickup Time: {stats['avg_pickup_time']:.1f}s")
        if stats['avg_delivery_time'] > 0:
            print(f"Average Delivery Time: {stats['avg_delivery_time']:.1f}s")
        
        # Final status
        print("-" * 60)
        if self.is_mission_complete():
            print("STATUS: Mission complete!")
        else:
            print("STATUS: Mission incomplete")
        print("=" * 60)
    
    def get_box_status(self, box_id: str) -> Optional[Dict]:
        """Get status of a specific box."""
        return self.box_states.get(box_id)
    
    def get_current_box_info(self) -> Optional[Dict]:
        """Get info about the currently held box."""
        if self.current_box_id:
            return self.get_box_status(self.current_box_id)
        return None 
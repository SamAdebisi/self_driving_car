"""
Car Simulation Module
Handles car physics, sensors, and movement
"""

import numpy as np
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.vector import Vector
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class Car(Widget):
    """
    Car widget with physics simulation and sensor system
    Industry-standard implementation with optimized sensor calculations
    """
    
    # Position and orientation
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    
    # Velocity
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    # Sensor positions (3 sensors for obstacle detection)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    
    # Sensor signals (distance measurements)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    
    def __init__(self, **kwargs):
        """Initialize car with default properties"""
        super(Car, self).__init__(**kwargs)
        self.sensor_range = 30
        self.sensor_angles = [0, 30, -30]
        self.sensor_width = 10
    
    def move(self, rotation: float, sand_map: np.ndarray, map_width: int, map_height: int):
        """
        Update car position and sensors
        
        Args:
            rotation: Rotation angle in degrees
            sand_map: 2D numpy array representing sand/obstacles
            map_width: Width of the map
            map_height: Height of the map
        """
        # Update position
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = (self.angle + self.rotation) % 360
        
        # Update sensor positions
        self._update_sensors()
        
        # Update sensor signals (detect obstacles)
        self._update_sensor_signals(sand_map, map_width, map_height)
    
    def _update_sensors(self):
        """Update sensor positions based on car angle"""
        # Sensor 1: Forward
        self.sensor1 = Vector(self.sensor_range, 0).rotate(self.angle) + self.pos
        
        # Sensor 2: Right (30 degrees)
        self.sensor2 = Vector(self.sensor_range, 0).rotate(
            (self.angle + self.sensor_angles[1]) % 360
        ) + self.pos
        
        # Sensor 3: Left (-30 degrees)
        self.sensor3 = Vector(self.sensor_range, 0).rotate(
            (self.angle + self.sensor_angles[2]) % 360
        ) + self.pos
    
    def _update_sensor_signals(
        self,
        sand_map: np.ndarray,
        map_width: int,
        map_height: int
    ):
        """
        Calculate sensor signals based on sand/obstacle detection
        
        Optimized for performance with vectorized operations
        """
        # Check boundary conditions first (faster than array access)
        sensors = [
            (self.sensor1_x, self.sensor1_y, 1),
            (self.sensor2_x, self.sensor2_y, 2),
            (self.sensor3_x, self.sensor3_y, 3),
        ]
        
        for sensor_x, sensor_y, sensor_num in sensors:
            # Boundary check
            if (sensor_x > map_width - self.sensor_width or
                sensor_x < self.sensor_width or
                sensor_y > map_height - self.sensor_width or
                sensor_y < self.sensor_width):
                # Hit boundary
                signal_value = 1.0
            else:
                # Check sand/obstacle density in sensor area
                x_start = max(0, int(sensor_x - self.sensor_width))
                x_end = min(map_width, int(sensor_x + self.sensor_width))
                y_start = max(0, int(sensor_y - self.sensor_width))
                y_end = min(map_height, int(sensor_y + self.sensor_width))
                
                # Vectorized calculation for better performance
                area_size = (x_end - x_start) * (y_end - y_start)
                if area_size > 0:
                    sand_density = np.sum(
                        sand_map[x_start:x_end, y_start:y_end]
                    ) / area_size
                    signal_value = min(1.0, sand_density)
                else:
                    signal_value = 0.0
            
            # Update signal property
            if sensor_num == 1:
                self.signal1 = signal_value
            elif sensor_num == 2:
                self.signal2 = signal_value
            elif sensor_num == 3:
                self.signal3 = signal_value
    
    def get_sensor_signals(self) -> Tuple[float, float, float]:
        """
        Get current sensor signals
        
        Returns:
            Tuple of (signal1, signal2, signal3)
        """
        return (self.signal1, self.signal2, self.signal3)
    
    def reset(self, center_x: float, center_y: float):
        """
        Reset car to initial position
        
        Args:
            center_x: X coordinate of center
            center_y: Y coordinate of center
        """
        self.center = (center_x, center_y)
        self.angle = 0
        self.velocity = Vector(6, 0)


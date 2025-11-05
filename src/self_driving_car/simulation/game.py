"""
Game Simulation Module
Main game loop and environment management
"""

import numpy as np
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.vector import Vector
from typing import Optional, Tuple
import logging

from ..rl.dqn import DQNAgent

logger = logging.getLogger(__name__)


class Ball1(Widget):
    """Visual indicator for sensor 1"""
    pass


class Ball2(Widget):
    """Visual indicator for sensor 2"""
    pass


class Ball3(Widget):
    """Visual indicator for sensor 3"""
    pass


class Game(Widget):
    """
    Main game widget managing the simulation environment
    Optimized for performance with efficient state management
    """
    
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    
    def __init__(
        self,
        agent: DQNAgent,
        sand_map: np.ndarray,
        config: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize game environment
        
        Args:
            agent: DQN agent for decision making
            sand_map: 2D numpy array representing sand/obstacles
            config: Configuration dictionary
            **kwargs: Additional widget arguments
        """
        super(Game, self).__init__(**kwargs)
        self.agent = agent
        self.sand_map = sand_map
        self.config = config or {}
        
        # Game state
        self.goal_x = self.config.get('goal_x', 20)
        self.goal_y = self.config.get('goal_y', 580)
        self.last_distance = float('inf')
        self.last_reward = 0.0
        self.scores = []
        
        # Rewards configuration
        self.rewards = {
            'sand_penalty': self.config.get('sand_penalty', -1.0),
            'wall_penalty': self.config.get('wall_penalty', -1.0),
            'distance_bonus': self.config.get('distance_bonus', 0.1),
            'base_penalty': self.config.get('base_penalty', -0.2),
            'goal_reward': self.config.get('goal_reward', 10.0),
        }
        
        # Car physics
        self.car_speed = self.config.get('car_speed', 6.0)
        self.sand_speed = self.config.get('sand_speed', 1.0)
        self.proximity_threshold = self.config.get('proximity_threshold', 100)
        
        logger.info("Game initialized")
    
    def serve_car(self):
        """Initialize car at center of map"""
        self.car.center = self.center
        self.car.velocity = Vector(self.car_speed, 0)
        self.last_distance = np.sqrt(
            (self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2
        )
    
    def update(self, dt: float):
        """
        Update game state (called every frame)
        
        Args:
            dt: Delta time since last update
        """
        # Get map dimensions
        map_width = int(self.width)
        map_height = int(self.height)
        
        # Ensure sand map is properly sized
        if (self.sand_map.shape[0] != map_width or
            self.sand_map.shape[1] != map_height):
            self.sand_map = np.zeros((map_width, map_height))
        
        # Calculate orientation to goal
        orientation = self._calculate_orientation()
        
        # Get sensor signals
        sensor_signals = self.car.get_sensor_signals()
        
        # Build state vector
        state = np.array([
            sensor_signals[0],
            sensor_signals[1],
            sensor_signals[2],
            orientation,
            -orientation
        ], dtype=np.float32)
        
        # Get action from agent
        action = self.agent.update(self.last_reward, state)
        
        # Map action to rotation
        action2rotation = [0, 20, -20]
        rotation = action2rotation[action]
        
        # Move car
        self.car.move(rotation, self.sand_map, map_width, map_height)
        
        # Update sensor visual indicators
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        
        # Calculate reward
        reward = self._calculate_reward()
        self.last_reward = reward
        
        # Update scores
        self.scores.append(self.agent.score())
        
        # Check goal proximity
        distance = np.sqrt(
            (self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2
        )
        
        if distance < self.proximity_threshold:
            # Reposition goal
            self.goal_x = map_width - self.goal_x
            self.goal_y = map_height - self.goal_y
            logger.debug(f"Goal repositioned to ({self.goal_x}, {self.goal_y})")
        
        self.last_distance = distance
    
    def _calculate_orientation(self) -> float:
        """
        Calculate orientation angle to goal
        
        Returns:
            Normalized orientation angle (-1 to 1)
        """
        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0
        return orientation
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current state
        
        Returns:
            Reward value
        """
        # Check if on sand (obstacle)
        car_x = int(self.car.x)
        car_y = int(self.car.y)
        
        if (0 <= car_x < self.sand_map.shape[0] and
            0 <= car_y < self.sand_map.shape[1] and
            self.sand_map[car_x, car_y] > 0):
            # On sand - penalty
            self.car.velocity = Vector(self.sand_speed, 0).rotate(self.car.angle)
            return self.rewards['sand_penalty']
        
        # Normal speed
        self.car.velocity = Vector(self.car_speed, 0).rotate(self.car.angle)
        
        # Check boundaries
        margin = 10
        if (self.car.x < margin or
            self.car.x > self.width - margin or
            self.car.y < margin or
            self.car.y > self.height - margin):
            # Hit wall - penalty
            self.car.x = max(margin, min(self.width - margin, self.car.x))
            self.car.y = max(margin, min(self.height - margin, self.car.y))
            return self.rewards['wall_penalty']
        
        # Calculate distance to goal
        distance = np.sqrt(
            (self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2
        )
        
        # Base penalty for moving
        reward = self.rewards['base_penalty']
        
        # Bonus for getting closer to goal
        if distance < self.last_distance:
            reward += self.rewards['distance_bonus']
        
        return reward
    
    def get_scores(self) -> list:
        """Get score history"""
        return self.scores


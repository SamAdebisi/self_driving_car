"""
Self-Driving Car Project
Industry-standard implementation with modern Deep Reinforcement Learning
"""

__version__ = "2.0.0"
__author__ = "Self-Driving Car Team"

from .rl.dqn import DQNAgent
from .simulation.car import Car
from .simulation.game import Game
from .simulation.app import CarApp

__all__ = [
    "DQNAgent",
    "Car",
    "Game",
    "CarApp",
]


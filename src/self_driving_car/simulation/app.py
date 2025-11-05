"""
Main Application Module
Kivy-based GUI application for self-driving car simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Line
from kivy.config import Config
from kivy.clock import Clock
from kivy.properties import NumericProperty
from typing import Optional, Tuple
import logging
from pathlib import Path

from .game import Game, Ball1, Ball2, Ball3
from .car import Car
from ..rl.dqn import DQNAgent
from ..utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Disable right-click to prevent red point drawing
Config.set('input', 'mouse', 'mouse,multitouch_demand')


class MyPaintWidget(Widget):
    """
    Painting widget for drawing sand/obstacles on the map
    Optimized for smooth drawing with density-based line width
    """
    
    def __init__(self, sand_map: np.ndarray, **kwargs):
        """
        Initialize paint widget
        
        Args:
            sand_map: Reference to sand map array
            **kwargs: Additional widget arguments
        """
        super(MyPaintWidget, self).__init__(**kwargs)
        self.sand_map = sand_map
        self.last_x = 0
        self.last_y = 0
        self.n_points = 0
        self.length = 0
    
    def on_touch_down(self, touch):
        """Handle touch down event"""
        with self.canvas:
            Color(0.8, 0.7, 0.0, 1.0)  # Sand color
            d = 10.0
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            self.last_x = int(touch.x)
            self.last_y = int(touch.y)
            self.n_points = 0
            self.length = 0
            
            # Update sand map
            x, y = int(touch.x), int(touch.y)
            if 0 <= x < self.sand_map.shape[0] and 0 <= y < self.sand_map.shape[1]:
                self.sand_map[x, y] = 1
    
    def on_touch_move(self, touch):
        """Handle touch move event with density-based line width"""
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            
            # Calculate density for line width
            self.length += np.sqrt(max((x - self.last_x)**2 + (y - self.last_y)**2, 2))
            self.n_points += 1.0
            density = self.n_points / (self.length + 1e-6)
            touch.ud['line'].width = int(20 * density + 1)
            
            # Update sand map in area
            x_start = max(0, x - 10)
            x_end = min(self.sand_map.shape[0], x + 10)
            y_start = max(0, y - 10)
            y_end = min(self.sand_map.shape[1], y + 10)
            self.sand_map[x_start:x_end, y_start:y_end] = 1
            
            self.last_x = x
            self.last_y = y


class CarApp(App):
    """
    Main application class
    Manages the simulation environment and UI
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize application
        
        Args:
            config_path: Path to configuration file
        """
        super(CarApp, self).__init__()
        self.config_loader = ConfigLoader(config_path)
        self.config_loader.validate()
        
        # Load configuration
        config = self.config_loader.config
        
        # Initialize sand map
        map_width = config['simulation']['map_width']
        map_height = config['simulation']['map_height']
        self.sand_map = np.zeros((map_width, map_height))
        
        # Initialize DQN agent
        network_config = config['network']
        rl_config = config['rl']
        
        self.agent = DQNAgent(
            input_size=network_config['input_size'],
            nb_action=network_config['output_size'],
            gamma=rl_config['gamma'],
            epsilon_start=rl_config['epsilon_start'],
            epsilon_end=rl_config['epsilon_end'],
            epsilon_decay=rl_config['epsilon_decay'],
            learning_rate=rl_config['learning_rate'],
            batch_size=rl_config['batch_size'],
            memory_capacity=rl_config['memory_capacity'],
            target_update_frequency=rl_config['target_update_frequency'],
            use_double_dqn=network_config['use_double_dqn'],
            use_dueling=network_config['use_dueling'],
            use_prioritized_replay=rl_config['use_prioritized_replay'],
            hidden_layers=network_config.get('hidden_layers', [128, 128, 64])
        )
        
        # Load saved model if exists
        try:
            self.agent.load()
        except Exception as e:
            logger.warning(f"Could not load saved model: {e}")
        
        # Game configuration
        self.game_config = {
            'goal_x': config['simulation']['goal']['initial_x'],
            'goal_y': config['simulation']['goal']['initial_y'],
            'car_speed': config['simulation']['car']['initial_speed'],
            'sand_speed': config['simulation']['car']['sand_speed'],
            'proximity_threshold': config['simulation']['goal']['proximity_threshold'],
            **config['rewards']
        }
        
        # UI components
        self.painter = None
        self.game = None
        
        logger.info("CarApp initialized")
    
    def build(self):
        """Build the application UI"""
        # Create game widget
        self.game = Game(
            agent=self.agent,
            sand_map=self.sand_map,
            config=self.game_config
        )
        self.game.serve_car()
        
        # Schedule update
        fps = self.config_loader.get('simulation.fps', 60)
        Clock.schedule_interval(self.game.update, 1.0 / fps)
        
        # Create paint widget
        self.painter = MyPaintWidget(self.sand_map)
        
        # Create buttons
        clearbtn = Button(text='Clear', size_hint=(None, None), size=(100, 50))
        savebtn = Button(text='Save', size_hint=(None, None), size=(100, 50), pos=(100, 0))
        loadbtn = Button(text='Load', size_hint=(None, None), size=(100, 50), pos=(200, 0))
        
        # Bind button events
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        
        # Add widgets
        self.game.add_widget(self.painter)
        self.game.add_widget(clearbtn)
        self.game.add_widget(savebtn)
        self.game.add_widget(loadbtn)
        
        return self.game
    
    def clear_canvas(self, obj):
        """Clear the sand map"""
        self.painter.canvas.clear()
        map_width, map_height = self.sand_map.shape
        self.sand_map = np.zeros((map_width, map_height))
        logger.info("Canvas cleared")
    
    def save(self, obj):
        """Save the model and plot scores"""
        logger.info("Saving model...")
        self.agent.save()
        
        # Plot scores if enabled
        if self.config_loader.get('training.visualization.plot_scores', True):
            scores = self.game.get_scores()
            if scores:
                plt.figure(figsize=(10, 6))
                plt.plot(scores)
                plt.title('Training Scores Over Time')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True)
                
                plot_dir = Path(__file__).parent.parent.parent.parent / "logs"
                plot_dir.mkdir(exist_ok=True)
                plot_path = plot_dir / "training_scores.png"
                plt.savefig(plot_path)
                logger.info(f"Score plot saved to {plot_path}")
                plt.show()
    
    def load(self, obj):
        """Load the saved model"""
        logger.info("Loading saved model...")
        try:
            self.agent.load()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


def main():
    """Main entry point"""
    import sys
    from ..utils.logger_setup import setup_logging
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Create and run app
    app = CarApp()
    app.run()


if __name__ == '__main__':
    main()


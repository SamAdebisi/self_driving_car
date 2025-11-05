# Self-Driving Car Simulation with Deep Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Industry-standard implementation of a self-driving car simulation using Deep Reinforcement Learning. The car learns to navigate a map with obstacles using state-of-the-art DQN algorithms including Double DQN, Dueling DQN, and Prioritized Experience Replay.

## Features

- 🚗 **Self-Driving Car Simulation**: Interactive environment with physics-based car movement
- 🧠 **Deep Reinforcement Learning**: Industry-standard DQN implementation
- 🎯 **Double DQN**: Reduces overestimation bias for more stable learning
- ⚡ **Dueling DQN**: Separates value and advantage estimation for better learning
- 📊 **Prioritized Experience Replay**: Samples important experiences more frequently
- 🎨 **Interactive UI**: Draw obstacles and watch the car learn to navigate
- ⚙️ **Configurable**: Easy to modify network structure and hyperparameters
- 🧪 **Comprehensive Testing**: Unit and integration tests with high coverage
- 📚 **Full Documentation**: Complete API and algorithm documentation

## Installation

### Prerequisites

- Python 3.8+
- pip

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd self_driving_car
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the simulation:**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

1. **Start the simulation:**
   ```bash
   python main.py
   ```

2. **Draw obstacles:**
   - Click and drag on the map to draw sand/obstacles
   - The car will learn to avoid these obstacles

3. **Save/Load progress:**
   - Click "Save" to save the trained model
   - Click "Load" to load a previously saved model
   - Click "Clear" to clear all obstacles

### Programmatic Usage

```python
from self_driving_car.rl.dqn import DQNAgent
from self_driving_car.simulation.app import CarApp

# Create agent with custom configuration
agent = DQNAgent(
    input_size=5,
    nb_action=3,
    use_double_dqn=True,
    use_dueling=True,
    use_prioritized_replay=True
)

# Run simulation
app = CarApp()
app.run()
```

## Architecture

### Project Structure

```
self_driving_car/
├── src/
│   └── self_driving_car/
│       ├── rl/              # Reinforcement Learning modules
│       ├── simulation/      # Simulation modules
│       └── utils/           # Utility modules
├── tests/                   # Test suite
├── docs/                    # Documentation
├── config/                  # Configuration files
└── main.py                 # Entry point
```

### System Components

- **DQN Agent**: Deep Q-Network with Double DQN, Dueling DQN, and Prioritized Replay
- **Car Simulation**: Physics-based car with sensor system
- **Game Environment**: Manages simulation state and rewards
- **Configuration System**: YAML-based configuration management
- **Logging System**: Structured logging for debugging and monitoring

## Configuration

Configuration is managed through YAML files. See `config/config.yaml` for all available options.

### Key Configuration Options

- **Network Architecture**: Layer sizes, activation functions, Dueling/Double DQN
- **RL Hyperparameters**: Learning rate, discount factor, epsilon decay
- **Simulation Settings**: Map size, car speed, sensor configuration
- **Rewards**: Penalties and bonuses for different behaviors

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage report
pytest --cov=src/self_driving_car --cov-report=html
```

### Test Coverage

The project includes comprehensive test coverage:
- **Unit Tests**: Individual components (networks, buffers, agent)
- **Integration Tests**: Complete workflows and pipelines
- **RL-Specific Tests**: Reinforcement learning algorithm validation

## Algorithms

### Deep Q-Network (DQN)

Standard DQN with experience replay and target network for stable learning.

### Double DQN

Reduces overestimation bias by using separate networks for action selection and evaluation.

### Dueling DQN

Separates value and advantage estimation for more efficient learning.

### Prioritized Experience Replay

Samples important experiences more frequently for faster convergence.

For detailed algorithm explanations, see [Theories Documentation](docs/THEORIES.md).

## Documentation

- **[Full Documentation](docs/README.md)**: Complete project documentation
- **[API Reference](docs/API.md)**: Detailed API documentation
- **[Theories](docs/THEORIES.md)**: Algorithm explanations and background

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Deep Reinforcement Learning algorithms (DQN, Double DQN, Dueling DQN, Prioritized Replay)
- Built with PyTorch, Kivy, and NumPy

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
3. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.
4. Schaul, T., et al. (2016). "Prioritized Experience Replay." ICLR.

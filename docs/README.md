# Self-Driving Car Project Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Algorithms](#algorithms)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [API Reference](#api-reference)
9. [Theories and Background](#theories-and-background)

## Overview

This project implements a self-driving car simulation using Deep Reinforcement Learning (DRL). The car learns to navigate a map with obstacles (sand) using state-of-the-art DQN algorithms including Double DQN, Dueling DQN, and Prioritized Experience Replay.

### Key Features

- **Industry-standard DQN implementation** with modern improvements
- **Double DQN**: Reduces overestimation bias in Q-value estimation
- **Dueling DQN**: Separates value and advantage estimation for better learning
- **Prioritized Experience Replay**: Samples important experiences more frequently
- **Configurable architecture**: Easy to modify network structure and hyperparameters
- **Comprehensive testing**: Unit and integration tests
- **Full documentation**: Complete API and algorithm documentation

## Architecture

### Project Structure

```
self_driving_car/
├── src/
│   └── self_driving_car/
│       ├── rl/              # Reinforcement Learning modules
│       │   ├── dqn.py       # Main DQN agent
│       │   ├── networks.py  # Neural network architectures
│       │   └── replay_buffer.py  # Experience replay buffers
│       ├── simulation/      # Simulation modules
│       │   ├── car.py      # Car physics and sensors
│       │   ├── game.py     # Game environment
│       │   └── app.py      # Main application
│       └── utils/          # Utility modules
│           ├── config_loader.py
│           └── logger_setup.py
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
├── config/                 # Configuration files
└── main.py                # Entry point
```

### System Architecture

```
┌─────────────────┐
│   Car (Sensor)  │
└────────┬────────┘
         │ State Vector
         ▼
┌─────────────────┐
│   DQN Agent     │
│  ┌───────────┐ │
│  │ Q-Network  │ │
│  │ (Main)     │ │
│  └─────┬──────┘ │
│        │        │
│  ┌─────▼──────┐ │
│  │Target Net  │ │
│  └────────────┘ │
└────────┬────────┘
         │ Action
         ▼
┌─────────────────┐
│  Environment    │
│  (Simulation)   │
└─────────────────┘
         │
         │ Reward
         ▼
┌─────────────────┐
│ Replay Buffer   │
│ (Experience)    │
└─────────────────┘
```

## Algorithms

### Deep Q-Network (DQN)

DQN is a value-based reinforcement learning algorithm that uses a neural network to approximate the Q-function. The Q-function estimates the expected cumulative reward for taking an action in a given state.

**Key Equations:**

1. **Q-Learning Update:**
   ```
   Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
   ```

2. **Loss Function:**
   ```
   L = E[(r + γ max Q(s', a') - Q(s, a))²]
   ```

3. **Target Q-Value:**
   ```
   y = r + γ max Q(s', a')
   ```

### Double DQN

Double DQN addresses the overestimation bias in standard DQN by using two separate networks:
- Main network selects the best action
- Target network evaluates that action

**Key Improvement:**
```
y = r + γ Q_target(s', argmax Q_main(s', a'))
```

This reduces overestimation and leads to more stable learning.

### Dueling DQN

Dueling DQN separates the Q-function into value and advantage streams:

**Architecture:**
```
State → Shared Layers → Value Stream → V(s)
                      → Advantage Stream → A(s, a)
```

**Q-Value Calculation:**
```
Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
```

This allows the network to learn which states are valuable independently of which actions are valuable.

### Prioritized Experience Replay

Prioritized Experience Replay samples experiences based on their TD-error priority:

**Priority Calculation:**
```
priority = |TD_error|^α
```

**Sampling Probability:**
```
P(i) = priority_i / Σ priority_j
```

**Importance Sampling Weight:**
```
w_i = (N * P(i))^(-β)
```

This ensures important experiences are learned from more frequently.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd self_driving_car
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   pytest tests/ -v
   ```

## Usage

### Basic Usage

Run the simulation:
```bash
python main.py
```

### Programmatic Usage

```python
from self_driving_car.rl.dqn import DQNAgent
from self_driving_car.simulation.app import CarApp

# Create agent
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

### Training

1. **Start the simulation**
2. **Draw obstacles** by clicking and dragging on the map
3. **Watch the car learn** to navigate around obstacles
4. **Save progress** by clicking the "Save" button
5. **Load previous model** by clicking the "Load" button

## Configuration

Configuration is managed through YAML files in the `config/` directory. See `config/config.yaml` for all available options.

### Key Configuration Options

- **Network Architecture**: Layer sizes, activation functions
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

# With coverage
pytest --cov=src/self_driving_car
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows and pipelines
- **RL-Specific Tests**: Test reinforcement learning algorithms

## API Reference

See [API Reference](API.md) for detailed API documentation.

## Theories and Background

See [Theories](THEORIES.md) for detailed explanations of algorithms, models, and frameworks used.


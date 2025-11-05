# API Reference

## Table of Contents

1. [RL Module](#rl-module)
2. [Simulation Module](#simulation-module)
3. [Utils Module](#utils-module)

## RL Module

### `DQNAgent`

Main Deep Q-Network agent with industry-standard improvements.

#### Constructor

```python
DQNAgent(
    input_size: int,
    nb_action: int,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    learning_rate: float = 0.0001,
    batch_size: int = 64,
    memory_capacity: int = 100000,
    target_update_frequency: int = 1000,
    use_double_dqn: bool = True,
    use_dueling: bool = True,
    use_prioritized_replay: bool = True,
    hidden_layers: Optional[list] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `input_size`: Size of input state vector
- `nb_action`: Number of possible actions
- `gamma`: Discount factor (0 ≤ γ ≤ 1)
- `epsilon_start`: Initial exploration rate
- `epsilon_end`: Final exploration rate
- `epsilon_decay`: Epsilon decay rate per step
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Batch size for training
- `memory_capacity`: Capacity of replay buffer
- `target_update_frequency`: Steps between target network updates
- `use_double_dqn`: Whether to use Double DQN
- `use_dueling`: Whether to use Dueling DQN architecture
- `use_prioritized_replay`: Whether to use prioritized experience replay
- `hidden_layers`: List of hidden layer sizes (default: [128, 128, 64])
- `device`: Device to use ('cpu' or 'cuda'), auto-detects if None

#### Methods

##### `select_action(state, training=True)`

Select action using epsilon-greedy policy.

**Parameters:**
- `state`: Current state (numpy array)
- `training`: Whether in training mode (affects exploration)

**Returns:**
- `int`: Selected action index

##### `remember(state, action, reward, next_state, done)`

Store experience in replay buffer.

**Parameters:**
- `state`: Current state
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state
- `done`: Whether episode ended

##### `learn()`

Train the agent on a batch of experiences.

**Returns:**
- `Optional[float]`: Loss value if training occurred, None otherwise

##### `update(reward, new_signal)`

Update agent with new observation and return action.

**Parameters:**
- `reward`: Reward received
- `new_signal`: New state observation

**Returns:**
- `int`: Selected action

##### `score()`

Calculate average reward over recent window.

**Returns:**
- `float`: Average reward score

##### `save(filepath=None)`

Save agent state to file.

**Parameters:**
- `filepath`: Path to save file (default: 'checkpoints/last_brain.pth')

##### `load(filepath=None)`

Load agent state from file.

**Parameters:**
- `filepath`: Path to load file (default: 'checkpoints/last_brain.pth')

### `DQN`

Standard Deep Q-Network architecture.

#### Constructor

```python
DQN(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = [128, 128, 64],
    activation: str = "relu"
)
```

**Parameters:**
- `input_size`: Size of input state vector
- `output_size`: Number of possible actions
- `hidden_layers`: List of hidden layer sizes
- `activation`: Activation function name ('relu', 'tanh', 'elu')

#### Methods

##### `forward(state)`

Forward pass through the network.

**Parameters:**
- `state`: Input state tensor (batch_size, input_size)

**Returns:**
- `torch.Tensor`: Q-values for each action (batch_size, output_size)

### `DuelingDQN`

Dueling DQN architecture with separate value and advantage streams.

#### Constructor

```python
DuelingDQN(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = [128, 128],
    activation: str = "relu"
)
```

**Parameters:**
- `input_size`: Size of input state vector
- `output_size`: Number of possible actions
- `hidden_layers`: List of hidden layer sizes for shared layers
- `activation`: Activation function name

#### Methods

##### `forward(state)`

Forward pass through dueling network.

**Parameters:**
- `state`: Input state tensor (batch_size, input_size)

**Returns:**
- `torch.Tensor`: Q-values for each action (batch_size, output_size)

### `ReplayBuffer`

Standard experience replay buffer.

#### Constructor

```python
ReplayBuffer(capacity: int)
```

**Parameters:**
- `capacity`: Maximum number of experiences to store

#### Methods

##### `push(state, action, reward, next_state, done)`

Add experience to buffer.

**Parameters:**
- `state`: Current state
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state
- `done`: Whether episode is done

##### `sample(batch_size)`

Sample batch of experiences.

**Parameters:**
- `batch_size`: Number of experiences to sample

**Returns:**
- `Tuple[torch.Tensor, ...]`: Tuple of (states, actions, rewards, next_states, dones)

### `PrioritizedReplayBuffer`

Prioritized experience replay buffer.

#### Constructor

```python
PrioritizedReplayBuffer(
    capacity: int,
    alpha: float = 0.6,
    beta: float = 0.4,
    beta_increment: float = 0.0001,
    min_priority: float = 0.01
)
```

**Parameters:**
- `capacity`: Maximum number of experiences to store
- `alpha`: Prioritization exponent (0 = uniform, 1 = fully prioritized)
- `beta`: Importance sampling exponent (0 = no correction, 1 = full correction)
- `beta_increment`: Amount to increment beta per sample
- `min_priority`: Minimum priority to avoid zero probabilities

#### Methods

##### `push(state, action, reward, next_state, done)`

Add experience to buffer with maximum priority.

##### `sample(batch_size)`

Sample batch of experiences based on priority.

**Returns:**
- `Tuple[torch.Tensor, ...]`: Tuple of (states, actions, rewards, next_states, dones, indices, weights)

##### `update_priorities(indices, td_errors)`

Update priorities based on TD-errors.

**Parameters:**
- `indices`: Indices of experiences to update
- `td_errors`: TD-errors for those experiences

## Simulation Module

### `Car`

Car widget with physics simulation and sensor system.

#### Properties

- `angle`: Current angle in degrees
- `rotation`: Rotation amount
- `velocity`: Velocity vector (velocity_x, velocity_y)
- `sensor1`, `sensor2`, `sensor3`: Sensor positions
- `signal1`, `signal2`, `signal3`: Sensor signals (obstacle detection)

#### Methods

##### `move(rotation, sand_map, map_width, map_height)`

Update car position and sensors.

**Parameters:**
- `rotation`: Rotation angle in degrees
- `sand_map`: 2D numpy array representing sand/obstacles
- `map_width`: Width of the map
- `map_height`: Height of the map

##### `get_sensor_signals()`

Get current sensor signals.

**Returns:**
- `Tuple[float, float, float]`: Tuple of (signal1, signal2, signal3)

##### `reset(center_x, center_y)`

Reset car to initial position.

**Parameters:**
- `center_x`: X coordinate of center
- `center_y`: Y coordinate of center

### `Game`

Main game widget managing the simulation environment.

#### Constructor

```python
Game(
    agent: DQNAgent,
    sand_map: np.ndarray,
    config: Optional[dict] = None,
    **kwargs
)
```

**Parameters:**
- `agent`: DQN agent for decision making
- `sand_map`: 2D numpy array representing sand/obstacles
- `config`: Configuration dictionary
- `**kwargs`: Additional widget arguments

#### Methods

##### `serve_car()`

Initialize car at center of map.

##### `update(dt)`

Update game state (called every frame).

**Parameters:**
- `dt`: Delta time since last update

##### `get_scores()`

Get score history.

**Returns:**
- `list`: List of scores

### `CarApp`

Main application class managing the simulation environment and UI.

#### Constructor

```python
CarApp(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path`: Path to configuration file

#### Methods

##### `build()`

Build the application UI.

**Returns:**
- `Widget`: Root widget

##### `clear_canvas(obj)`

Clear the sand map.

##### `save(obj)`

Save the model and plot scores.

##### `load(obj)`

Load the saved model.

## Utils Module

### `ConfigLoader`

Loads and manages configuration from YAML files.

#### Constructor

```python
ConfigLoader(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path`: Path to configuration file. If None, uses default.

#### Methods

##### `load()`

Load configuration from YAML file.

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

##### `get(key, default=None)`

Get configuration value by key (supports nested keys with dots).

**Parameters:**
- `key`: Configuration key (e.g., 'rl.gamma' or 'network.hidden_layers')
- `default`: Default value if key not found

**Returns:**
- `Any`: Configuration value

##### `validate()`

Validate configuration values.

**Returns:**
- `bool`: True if valid

### `setup_logging(log_dir=None, log_level="INFO", log_to_file=True, log_to_console=True)`

Set up logging configuration.

**Parameters:**
- `log_dir`: Directory for log files. If None, uses 'logs' in project root
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_to_file`: Whether to log to file
- `log_to_console`: Whether to log to console

**Returns:**
- `logging.Logger`: Configured logger


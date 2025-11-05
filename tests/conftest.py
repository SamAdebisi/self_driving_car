"""
Pytest configuration and fixtures
Provides common test fixtures and setup
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from self_driving_car.rl.dqn import DQNAgent
from self_driving_car.rl.networks import DQN, DuelingDQN
from self_driving_car.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from self_driving_car.utils.config_loader import ConfigLoader


@pytest.fixture
def sample_state():
    """Sample state vector for testing"""
    return np.array([0.5, 0.3, 0.7, 0.2, -0.2], dtype=np.float32)


@pytest.fixture
def sample_batch():
    """Sample batch of experiences for testing"""
    batch_size = 32
    state_size = 5
    return {
        'states': np.random.randn(batch_size, state_size).astype(np.float32),
        'actions': np.random.randint(0, 3, batch_size),
        'rewards': np.random.randn(batch_size).astype(np.float32),
        'next_states': np.random.randn(batch_size, state_size).astype(np.float32),
        'dones': np.random.choice([True, False], batch_size),
    }


@pytest.fixture
def dqn_agent():
    """Create a DQN agent for testing"""
    return DQNAgent(
        input_size=5,
        nb_action=3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        learning_rate=0.0001,
        batch_size=32,
        memory_capacity=10000,
        use_double_dqn=True,
        use_dueling=True,
        use_prioritized_replay=False,
        device='cpu'
    )


@pytest.fixture
def dqn_agent_prioritized():
    """Create a DQN agent with prioritized replay for testing"""
    return DQNAgent(
        input_size=5,
        nb_action=3,
        gamma=0.99,
        use_prioritized_replay=True,
        device='cpu'
    )


@pytest.fixture
def replay_buffer():
    """Create a standard replay buffer for testing"""
    return ReplayBuffer(capacity=1000)


@pytest.fixture
def prioritized_replay_buffer():
    """Create a prioritized replay buffer for testing"""
    return PrioritizedReplayBuffer(capacity=1000)


@pytest.fixture
def dqn_network():
    """Create a DQN network for testing"""
    return DQN(input_size=5, output_size=3, hidden_layers=[64, 32])


@pytest.fixture
def dueling_dqn_network():
    """Create a Dueling DQN network for testing"""
    return DuelingDQN(input_size=5, output_size=3, hidden_layers=[64, 32])


@pytest.fixture
def sand_map():
    """Create a sample sand map for testing"""
    return np.zeros((100, 100), dtype=np.float32)


@pytest.fixture
def config_loader(tmp_path):
    """Create a temporary config file and loader"""
    config_content = """
network:
  input_size: 5
  output_size: 3
  hidden_layers: [64, 32]
  use_dueling: true
  use_double_dqn: true

rl:
  gamma: 0.99
  learning_rate: 0.0001
  batch_size: 32
  memory_capacity: 10000

simulation:
  fps: 60
  map_width: 800
  map_height: 600
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return ConfigLoader(str(config_file))


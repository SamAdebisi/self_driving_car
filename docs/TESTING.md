# Testing Guide

## Overview

This project includes comprehensive testing with unit tests, integration tests, and RL-specific tests.

## Running Tests

### Prerequisites

Install dependencies first:
```bash
pip install -r requirements.txt
```

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

# Specific test file
pytest tests/unit/test_networks.py
```

### Run with Coverage

```bash
# Terminal coverage report
pytest --cov=src/self_driving_car --cov-report=term-missing

# HTML coverage report
pytest --cov=src/self_driving_car --cov-report=html
open htmlcov/index.html  # View in browser
```

### Run Verbose Output

```bash
pytest -v
```

### Run Specific Test

```bash
pytest tests/unit/test_networks.py::TestDQN::test_dqn_forward -v
```

## Test Structure

### Unit Tests

Unit tests test individual components in isolation:

- **test_networks.py**: Tests for neural network architectures (DQN, Dueling DQN)
- **test_replay_buffer.py**: Tests for experience replay buffers (standard and prioritized)
- **test_dqn_agent.py**: Tests for DQN agent functionality

### Integration Tests

Integration tests test complete workflows:

- **test_rl_pipeline.py**: Tests complete RL training pipeline
- **test_config_loader.py**: Tests configuration management

### Test Fixtures

Common test fixtures are defined in `tests/conftest.py`:

- `dqn_agent`: DQN agent for testing
- `dqn_agent_prioritized`: DQN agent with prioritized replay
- `replay_buffer`: Standard replay buffer
- `prioritized_replay_buffer`: Prioritized replay buffer
- `dqn_network`: DQN network
- `dueling_dqn_network`: Dueling DQN network
- `sample_state`: Sample state vector
- `sample_batch`: Sample batch of experiences
- `sand_map`: Sample sand map
- `config_loader`: Configuration loader

## Writing Tests

### Example Unit Test

```python
def test_dqn_forward():
    """Test DQN forward pass"""
    network = DQN(input_size=5, output_size=3)
    state = torch.randn(1, 5)
    q_values = network(state)
    
    assert q_values.shape == (1, 3)
    assert not torch.isnan(q_values).any()
```

### Example Integration Test

```python
def test_complete_training_episode(dqn_agent):
    """Test a complete training episode"""
    state = np.random.randn(5).astype(np.float32)
    
    for step in range(200):
        action = dqn_agent.select_action(state, training=True)
        next_state = state + np.random.randn(5) * 0.1
        reward = -np.linalg.norm(next_state)
        done = step == 199
        
        dqn_agent.remember(state, action, reward, next_state, done)
        
        if len(dqn_agent.memory) > dqn_agent.batch_size:
            loss = dqn_agent.learn()
            assert loss is not None
    
    assert len(dqn_agent.memory) > 0
```

## Test Coverage

The project aims for high test coverage:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test complete workflows
- **RL-Specific Tests**: Test reinforcement learning algorithms

Run coverage report:
```bash
pytest --cov=src/self_driving_car --cov-report=html
```

## Continuous Integration

Tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=src/self_driving_car --cov-report=xml
```

## Troubleshooting

### Import Errors

If you get import errors, ensure:
1. Dependencies are installed: `pip install -r requirements.txt`
2. Python path includes `src/`: `export PYTHONPATH="${PYTHONPATH}:src"`

### Kivy Import Errors

Kivy is required for simulation tests. If tests fail with Kivy import errors:
- Install Kivy: `pip install kivy`
- Or skip simulation tests: `pytest tests/unit/test_networks.py` (doesn't require Kivy)

### GPU Tests

By default, tests run on CPU. To test GPU functionality:
- Ensure CUDA is available
- Set device in test: `device='cuda'`


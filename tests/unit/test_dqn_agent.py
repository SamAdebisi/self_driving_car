"""
Unit tests for DQN agent
"""

import pytest
import numpy as np
import torch

from self_driving_car.rl.dqn import DQNAgent


class TestDQNAgent:
    """Test DQN agent"""
    
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            device='cpu'
        )
        assert agent.input_size == 5
        assert agent.nb_action == 3
        assert agent.epsilon == agent.epsilon_start
    
    def test_dqn_agent_select_action(self):
        """Test action selection"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            epsilon_start=0.0,  # No exploration
            device='cpu'
        )
        
        state = np.array([0.5, 0.3, 0.7, 0.2, -0.2], dtype=np.float32)
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < 3
    
    def test_dqn_agent_select_action_exploration(self):
        """Test action selection with exploration"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            epsilon_start=1.0,  # Always explore
            device='cpu'
        )
        
        state = np.array([0.5, 0.3, 0.7, 0.2, -0.2], dtype=np.float32)
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < 3
    
    def test_dqn_agent_remember(self):
        """Test storing experiences"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            memory_capacity=100,
            device='cpu'
        )
        
        state = np.random.randn(5).astype(np.float32)
        next_state = np.random.randn(5).astype(np.float32)
        
        agent.remember(state, 1, 0.5, next_state, False)
        
        assert len(agent.memory) == 1
    
    def test_dqn_agent_learn(self):
        """Test learning from experiences"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Add experiences to buffer
        for i in range(100):
            state = np.random.randn(5).astype(np.float32)
            next_state = np.random.randn(5).astype(np.float32)
            agent.remember(state, i % 3, float(i), next_state, False)
        
        # Learn
        loss = agent.learn()
        
        assert loss is not None
        assert loss >= 0
    
    def test_dqn_agent_learn_insufficient_experiences(self):
        """Test learning with insufficient experiences"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            batch_size=32,
            device='cpu'
        )
        
        # Add few experiences
        for i in range(10):
            state = np.random.randn(5).astype(np.float32)
            agent.remember(state, 0, 0.0, state, False)
        
        # Should return None (not enough experiences)
        loss = agent.learn()
        assert loss is None
    
    def test_dqn_agent_update(self):
        """Test update method"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(100):
            state = np.random.randn(5).astype(np.float32)
            agent.remember(state, 0, 0.0, state, False)
        
        # Update
        new_state = np.random.randn(5).astype(np.float32)
        action = agent.update(0.5, new_state)
        
        assert 0 <= action < 3
    
    def test_dqn_agent_epsilon_decay(self):
        """Test epsilon decay"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.9,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        initial_epsilon = agent.epsilon
        
        # Fill buffer and learn multiple times
        for i in range(100):
            state = np.random.randn(5).astype(np.float32)
            agent.remember(state, 0, 0.0, state, False)
        
        for _ in range(10):
            agent.learn()
        
        # Epsilon should have decreased
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_end
    
    def test_dqn_agent_double_dqn(self):
        """Test Double DQN functionality"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            use_double_dqn=True,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(100):
            state = np.random.randn(5).astype(np.float32)
            agent.remember(state, 0, 0.0, state, False)
        
        # Learn
        loss = agent.learn()
        assert loss is not None
    
    def test_dqn_agent_prioritized_replay(self):
        """Test prioritized experience replay"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            use_prioritized_replay=True,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(100):
            state = np.random.randn(5).astype(np.float32)
            agent.remember(state, 0, 0.0, state, False)
        
        # Learn
        loss = agent.learn()
        assert loss is not None
    
    def test_dqn_agent_save_load(self, tmp_path):
        """Test saving and loading agent"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            device='cpu'
        )
        
        # Save
        save_path = tmp_path / "test_agent.pth"
        agent.save(str(save_path))
        assert save_path.exists()
        
        # Load
        agent2 = DQNAgent(
            input_size=5,
            nb_action=3,
            device='cpu'
        )
        agent2.load(str(save_path))
        
        # Check that networks have same weights
        for param1, param2 in zip(
            agent.q_network.parameters(),
            agent2.q_network.parameters()
        ):
            assert torch.allclose(param1, param2)
    
    def test_dqn_agent_target_network_update(self):
        """Test target network update"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            target_update_frequency=10,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(100):
            state = np.random.randn(5).astype(np.float32)
            agent.remember(state, 0, 0.0, state, False)
        
        initial_steps = agent.steps
        
        # Learn multiple times
        for _ in range(15):
            agent.learn()
        
        # Target network should have been updated
        assert agent.steps > initial_steps


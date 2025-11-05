"""
Unit tests for experience replay buffers
"""

import pytest
import numpy as np
import torch

from self_driving_car.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class TestReplayBuffer:
    """Test standard replay buffer"""
    
    def test_replay_buffer_initialization(self):
        """Test replay buffer initialization"""
        buffer = ReplayBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert len(buffer) == 0
    
    def test_replay_buffer_push(self):
        """Test adding experiences to buffer"""
        buffer = ReplayBuffer(capacity=100)
        state = np.array([0.5, 0.3, 0.7, 0.2, -0.2])
        action = 1
        reward = 0.5
        next_state = np.array([0.6, 0.4, 0.6, 0.3, -0.1])
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1
    
    def test_replay_buffer_capacity(self):
        """Test that buffer respects capacity"""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(20):
            state = np.random.randn(5)
            buffer.push(state, 0, 0.0, state, False)
        
        assert len(buffer) == 10
    
    def test_replay_buffer_sample(self):
        """Test sampling from buffer"""
        buffer = ReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(50):
            state = np.random.randn(5)
            buffer.push(state, i % 3, float(i), state, False)
        
        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, 5)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 5)
        assert dones.shape == (batch_size,)
    
    def test_replay_buffer_sample_small_buffer(self):
        """Test sampling when buffer is smaller than batch size"""
        buffer = ReplayBuffer(capacity=100)
        
        # Add fewer experiences than batch size
        for i in range(10):
            state = np.random.randn(5)
            buffer.push(state, 0, 0.0, state, False)
        
        # Should still work
        states, actions, rewards, next_states, dones = buffer.sample(32)
        assert states.shape[0] == 10  # Should use available experiences


class TestPrioritizedReplayBuffer:
    """Test prioritized replay buffer"""
    
    def test_prioritized_replay_buffer_initialization(self):
        """Test prioritized replay buffer initialization"""
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            alpha=0.6,
            beta=0.4
        )
        assert buffer.capacity == 1000
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer) == 0
    
    def test_prioritized_replay_buffer_push(self):
        """Test adding experiences to prioritized buffer"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.array([0.5, 0.3, 0.7, 0.2, -0.2])
        
        buffer.push(state, 1, 0.5, state, False)
        assert len(buffer) == 1
    
    def test_prioritized_replay_buffer_sample(self):
        """Test sampling from prioritized buffer"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(5)
            buffer.push(state, i % 3, float(i), state, False)
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            buffer.sample(32)
        
        assert states.shape == (32, 5)
        assert len(indices) == 32
        assert len(weights) == 32
        assert weights.min() >= 0
        assert weights.max() <= 1
    
    def test_prioritized_replay_buffer_update_priorities(self):
        """Test updating priorities"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(5)
            buffer.push(state, 0, 0.0, state, False)
        
        # Sample and update priorities
        states, actions, rewards, next_states, dones, indices, weights = \
            buffer.sample(32)
        
        td_errors = np.random.randn(32)
        buffer.update_priorities(indices, td_errors)
        
        # Should not raise error
        assert True
    
    def test_prioritized_replay_buffer_beta_increment(self):
        """Test that beta increments correctly"""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            beta=0.4,
            beta_increment=0.01
        )
        
        initial_beta = buffer.beta
        
        # Sample to increment beta
        for i in range(10):
            state = np.random.randn(5)
            buffer.push(state, 0, 0.0, state, False)
        
        buffer.sample(5)
        
        # Beta should have increased (but capped at 1.0)
        assert buffer.beta >= initial_beta
        assert buffer.beta <= 1.0


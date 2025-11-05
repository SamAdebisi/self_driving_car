"""
Integration tests for reinforcement learning pipeline
Tests the complete RL workflow from state to action to learning
"""

import pytest
import numpy as np

from self_driving_car.rl.dqn import DQNAgent


class TestRLPipeline:
    """Test complete RL pipeline"""
    
    def test_complete_training_episode(self):
        """Test a complete training episode"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Simulate episode
        state = np.random.randn(5).astype(np.float32)
        
        for step in range(200):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Simulate environment
            next_state = state + np.random.randn(5) * 0.1
            reward = -np.linalg.norm(next_state)  # Reward for staying near origin
            done = step == 199
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    assert loss >= 0
            
            # Update state
            state = next_state
        
        # Check that agent learned something
        assert len(agent.memory) > 0
        assert agent.epsilon < agent.epsilon_start
    
    def test_double_dqn_pipeline(self):
        """Test Double DQN pipeline"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            use_double_dqn=True,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer and train
        for i in range(200):
            state = np.random.randn(5).astype(np.float32)
            next_state = np.random.randn(5).astype(np.float32)
            agent.remember(state, i % 3, float(i), next_state, False)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    assert loss >= 0
    
    def test_prioritized_replay_pipeline(self):
        """Test prioritized replay pipeline"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            use_prioritized_replay=True,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer and train
        for i in range(200):
            state = np.random.randn(5).astype(np.float32)
            next_state = np.random.randn(5).astype(np.float32)
            agent.remember(state, i % 3, float(i), next_state, False)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    assert loss >= 0
    
    def test_dueling_dqn_pipeline(self):
        """Test Dueling DQN pipeline"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            use_dueling=True,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Fill buffer and train
        for i in range(200):
            state = np.random.randn(5).astype(np.float32)
            next_state = np.random.randn(5).astype(np.float32)
            agent.remember(state, i % 3, float(i), next_state, False)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    assert loss >= 0
    
    def test_all_features_combined(self):
        """Test all RL features combined"""
        agent = DQNAgent(
            input_size=5,
            nb_action=3,
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True,
            batch_size=32,
            memory_capacity=1000,
            device='cpu'
        )
        
        # Simulate training
        for i in range(300):
            state = np.random.randn(5).astype(np.float32)
            next_state = np.random.randn(5).astype(np.float32)
            reward = -np.linalg.norm(next_state)
            
            agent.remember(state, i % 3, reward, next_state, False)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    assert loss >= 0
        
        # Check training progress
        assert agent.epsilon < agent.epsilon_start
        assert len(agent.memory) > agent.batch_size


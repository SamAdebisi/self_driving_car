"""
Unit tests for neural network architectures
"""

import pytest
import torch
import numpy as np

from self_driving_car.rl.networks import DQN, DuelingDQN


class TestDQN:
    """Test standard DQN network"""
    
    def test_dqn_initialization(self):
        """Test DQN network initialization"""
        network = DQN(input_size=5, output_size=3, hidden_layers=[64, 32])
        assert network.input_size == 5
        assert network.output_size == 3
        assert len(network.hidden_layers) == 2
    
    def test_dqn_forward(self):
        """Test DQN forward pass"""
        network = DQN(input_size=5, output_size=3)
        state = torch.randn(1, 5)
        q_values = network(state)
        
        assert q_values.shape == (1, 3)
        assert not torch.isnan(q_values).any()
    
    def test_dqn_batch_forward(self):
        """Test DQN forward pass with batch"""
        network = DQN(input_size=5, output_size=3)
        batch_size = 32
        states = torch.randn(batch_size, 5)
        q_values = network(states)
        
        assert q_values.shape == (batch_size, 3)
    
    def test_dqn_activation_functions(self):
        """Test different activation functions"""
        activations = ['relu', 'tanh', 'elu']
        for activation in activations:
            network = DQN(
                input_size=5,
                output_size=3,
                activation=activation
            )
            state = torch.randn(1, 5)
            q_values = network(state)
            assert q_values.shape == (1, 3)


class TestDuelingDQN:
    """Test Dueling DQN network"""
    
    def test_dueling_dqn_initialization(self):
        """Test Dueling DQN network initialization"""
        network = DuelingDQN(input_size=5, output_size=3, hidden_layers=[64, 32])
        assert network.input_size == 5
        assert network.output_size == 3
    
    def test_dueling_dqn_forward(self):
        """Test Dueling DQN forward pass"""
        network = DuelingDQN(input_size=5, output_size=3)
        state = torch.randn(1, 5)
        q_values = network(state)
        
        assert q_values.shape == (1, 3)
        assert not torch.isnan(q_values).any()
    
    def test_dueling_dqn_batch_forward(self):
        """Test Dueling DQN forward pass with batch"""
        network = DuelingDQN(input_size=5, output_size=3)
        batch_size = 32
        states = torch.randn(batch_size, 5)
        q_values = network(states)
        
        assert q_values.shape == (batch_size, 3)
    
    def test_dueling_dqn_value_advantage_separation(self):
        """Test that value and advantage streams are properly combined"""
        network = DuelingDQN(input_size=5, output_size=3)
        state = torch.randn(1, 5)
        q_values = network(state)
        
        # Q-values should be reasonable (not all zeros or same)
        assert not torch.allclose(q_values, torch.zeros_like(q_values))
        assert q_values.std() > 0.01


class TestNetworkIntegration:
    """Integration tests for networks"""
    
    def test_network_gradient_flow(self):
        """Test that gradients flow through networks"""
        network = DQN(input_size=5, output_size=3)
        state = torch.randn(1, 5, requires_grad=True)
        q_values = network(state)
        loss = q_values.mean()
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in network.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients
    
    def test_network_parameters_count(self):
        """Test that networks have reasonable number of parameters"""
        network = DQN(input_size=5, output_size=3, hidden_layers=[64, 32])
        param_count = sum(p.numel() for p in network.parameters())
        
        # Should have at least some parameters
        assert param_count > 100
        # Should not have unreasonably many
        assert param_count < 100000


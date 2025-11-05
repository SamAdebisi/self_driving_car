"""
Neural Network Architectures for Deep Q-Learning
Implements standard DQN and Dueling DQN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """
    Standard Deep Q-Network (DQN) architecture
    Industry-standard implementation with configurable layers
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [128, 128, 64],
        activation: str = "relu"
    ):
        """
        Initialize DQN network
        
        Args:
            input_size: Size of input state vector
            output_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
            activation: Activation function name ('relu', 'tanh', 'elu')
        """
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # Select activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
        }
        self.activation = activations.get(
            activation.lower(),
            nn.ReLU()
        )
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: Input state tensor (batch_size, input_size)
            
        Returns:
            Q-values for each action (batch_size, output_size)
        """
        return self.network(state)


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture
    Separates value and advantage estimation for better learning
    Based on: "Dueling Network Architectures for Deep Reinforcement Learning"
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [128, 128],
        activation: str = "relu"
    ):
        """
        Initialize Dueling DQN network
        
        Args:
            input_size: Size of input state vector
            output_size: Number of possible actions
            hidden_layers: List of hidden layer sizes for shared layers
            activation: Activation function name
        """
        super(DuelingDQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
        }
        self.activation = activations.get(
            activation.lower(),
            nn.ReLU()
        )
        
        # Shared feature layers
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            shared_layers.append(self.activation)
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream (estimates state value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 128),
            self.activation,
            nn.Linear(128, 1)
        )
        
        # Advantage stream (estimates action advantages A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 128),
            self.activation,
            nn.Linear(128, output_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling network
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        
        Args:
            state: Input state tensor (batch_size, input_size)
            
        Returns:
            Q-values for each action (batch_size, output_size)
        """
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This ensures the value function is properly separated
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


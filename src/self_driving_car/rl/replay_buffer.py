"""
Experience Replay Buffers
Implements standard and prioritized experience replay
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from collections import namedtuple
import random
import logging

logger = logging.getLogger(__name__)

# Experience tuple
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayBuffer:
    """
    Standard Experience Replay Buffer
    Stores and samples experiences uniformly
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory: List[Experience] = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Experience(
            state, action, reward, next_state, done
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        
        batch = random.sample(self.memory, batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on TD-error priority
    Based on: "Prioritized Experience Replay" (Schaul et al., 2016)
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.0001,
        min_priority: float = 0.01
    ):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increment beta per sample
            min_priority: Minimum priority to avoid zero probabilities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.min_priority = min_priority
        
        # Sum-tree for efficient priority sampling
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        
        self.sum_tree = np.zeros(2 * self.tree_size - 1)
        self.min_tree = np.full(2 * self.tree_size - 1, float('inf'))
        
        self.memory: List[Experience] = []
        self.position = 0
        self.max_priority = 1.0
    
    def _propagate(self, idx: int, change: float):
        """Update priority tree from leaf to root"""
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        self.min_tree[parent] = min(
            self.min_tree[2 * parent + 1],
            self.min_tree[2 * parent + 2]
        )
        if parent != 0:
            self._propagate(parent, change)
    
    def _update(self, idx: int, priority: float):
        """Update priority at leaf node"""
        idx += self.tree_size - 1
        change = priority - self.sum_tree[idx]
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority
        self._propagate(idx, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index from priority tree"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.sum_tree):
            return idx
        
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add experience to buffer with maximum priority
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Experience(
            state, action, reward, next_state, done
        )
        
        # Set maximum priority for new experiences
        priority = self.max_priority ** self.alpha
        self._update(self.position, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of experiences based on priority
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        
        indices = []
        priorities = []
        segment = self.sum_tree[0] / batch_size
        
        # Sample based on priority
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx = self._retrieve(0, s)
            indices.append(idx)
            priorities.append(self.sum_tree[idx])
        
        # Calculate importance sampling weights
        probabilities = np.array(priorities) / self.sum_tree[0]
        weights = (len(self.memory) * probabilities) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract experiences
        batch = [self.memory[idx] for idx in indices]
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD-errors
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD-errors for those experiences
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self._update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.memory)


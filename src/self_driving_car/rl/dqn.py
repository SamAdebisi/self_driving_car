"""
Deep Q-Network (DQN) Agent
Industry-standard implementation with Double DQN, Dueling DQN, and Prioritized Experience Replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
import os
from pathlib import Path

from .networks import DQN, DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network Agent with industry-standard improvements:
    - Double DQN: Reduces overestimation of Q-values
    - Dueling DQN: Separates value and advantage estimation
    - Prioritized Experience Replay: Samples important experiences more frequently
    - Epsilon-greedy exploration with decay
    - Target network for stable learning
    """
    
    def __init__(
        self,
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
    ):
        """
        Initialize DQN Agent
        
        Args:
            input_size: Size of input state vector
            nb_action: Number of possible actions
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per step
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            memory_capacity: Capacity of replay buffer
            target_update_frequency: Steps between target network updates
            use_double_dqn: Whether to use Double DQN
            use_dueling: Whether to use Dueling DQN architecture
            use_prioritized_replay: Whether to use prioritized experience replay
            hidden_layers: List of hidden layer sizes (default: [128, 128, 64])
            device: Device to use ('cpu' or 'cuda'), auto-detects if None
        """
        self.input_size = input_size
        self.nb_action = nb_action
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.use_double_dqn = use_double_dqn
        self.steps = 0
        
        # Device selection
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Network architecture
        if hidden_layers is None:
            hidden_layers = [128, 128, 64]
        
        # Create main and target networks
        if use_dueling:
            self.q_network = DuelingDQN(
                input_size, nb_action, hidden_layers
            ).to(self.device)
            self.target_network = DuelingDQN(
                input_size, nb_action, hidden_layers
            ).to(self.device)
        else:
            self.q_network = DQN(
                input_size, nb_action, hidden_layers
            ).to(self.device)
            self.target_network = DQN(
                input_size, nb_action, hidden_layers
            ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_capacity)
            self.use_prioritized_replay = True
        else:
            self.memory = ReplayBuffer(memory_capacity)
            self.use_prioritized_replay = False
        
        # Training statistics
        self.reward_window = []
        self.loss_history = []
        self.last_state = None
        self.last_action = None
        
        logger.info("DQN Agent initialized")
        logger.info(f"  - Double DQN: {use_double_dqn}")
        logger.info(f"  - Dueling DQN: {use_dueling}")
        logger.info(f"  - Prioritized Replay: {use_prioritized_replay}")
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Random exploration
            return np.random.randint(0, self.nb_action)
        
        # Exploitation: select best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(1).item()
        
        return action
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        """
        Train the agent on a batch of experiences
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        if self.use_prioritized_replay:
            (
                states, actions, rewards, next_states, dones,
                indices, weights
            ) = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size)
            weights = None
            indices = None
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        
        # Next Q-values using target network
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select actions,
                # target network to evaluate them
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN: use target network for both
                next_q_values = self.target_network(next_states).max(1)[0]
            
            # Target Q-values
            target_q_values = rewards + (
                self.gamma * next_q_values * ~dones
            )
        
        # Compute loss
        if self.use_prioritized_replay:
            # Weighted loss for prioritized replay
            td_errors = current_q_values - target_q_values
            loss = (weights * F.smooth_l1_loss(
                current_q_values, target_q_values, reduction='none'
            )).mean()
        else:
            # Standard loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            with torch.no_grad():
                td_errors = (current_q_values - target_q_values).cpu().numpy()
            self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug(f"Target network updated at step {self.steps}")
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update(
        self,
        reward: float,
        new_signal: np.ndarray
    ) -> int:
        """
        Update agent with new observation and return action
        
        Args:
            reward: Reward received
            new_signal: New state observation
            
        Returns:
            Selected action
        """
        new_state = new_signal.astype(np.float32)
        
        # Store experience if we have a previous state
        if self.last_state is not None:
            self.remember(
                self.last_state,
                self.last_action,
                reward,
                new_state,
                False  # Episode done flag (could be enhanced)
            )
        
        # Select action
        action = self.select_action(new_state, training=True)
        
        # Train
        self.learn()
        
        # Update state
        self.last_state = new_state
        self.last_action = action
        
        return action
    
    def score(self) -> float:
        """
        Calculate average reward over recent window
        
        Returns:
            Average reward score
        """
        if len(self.reward_window) == 0:
            return 0.0
        return sum(self.reward_window) / len(self.reward_window)
    
    def save(self, filepath: Optional[str] = None):
        """
        Save agent state to file
        
        Args:
            filepath: Path to save file (default: 'checkpoints/last_brain.pth')
        """
        if filepath is None:
            checkpoint_dir = Path(__file__).parent.parent.parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            filepath = checkpoint_dir / "last_brain.pth"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'input_size': self.input_size,
            'nb_action': self.nb_action,
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """
        Load agent state from file
        
        Args:
            filepath: Path to load file (default: 'checkpoints/last_brain.pth')
        """
        if filepath is None:
            checkpoint_dir = Path(__file__).parent.parent.parent.parent / "checkpoints"
            filepath = checkpoint_dir / "last_brain.pth"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"No checkpoint found at {filepath}")
            return
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(
                checkpoint['q_network_state_dict']
            )
            self.target_network.load_state_dict(
                checkpoint['target_network_state_dict']
            )
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.steps = checkpoint.get('steps', 0)
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise


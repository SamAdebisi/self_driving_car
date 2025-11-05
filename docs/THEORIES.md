# Theoretical Background and Algorithms

## Table of Contents

1. [Reinforcement Learning Fundamentals](#reinforcement-learning-fundamentals)
2. [Deep Q-Network (DQN)](#deep-q-network-dqn)
3. [Double DQN](#double-dqn)
4. [Dueling DQN](#dueling-dqn)
5. [Prioritized Experience Replay](#prioritized-experience-replay)
6. [Neural Network Architectures](#neural-network-architectures)
7. [Tools and Frameworks](#tools-and-frameworks)

## Reinforcement Learning Fundamentals

### Markov Decision Process (MDP)

Reinforcement Learning problems are typically formalized as Markov Decision Processes (MDPs):

**Components:**
- **State Space (S)**: Set of all possible states
- **Action Space (A)**: Set of all possible actions
- **Transition Probability (P)**: P(s'|s, a) - probability of transitioning to state s' from state s by taking action a
- **Reward Function (R)**: R(s, a, s') - reward received when transitioning from s to s' via action a
- **Discount Factor (γ)**: Determines importance of future rewards (0 ≤ γ ≤ 1)

### Policy

A policy π is a mapping from states to actions:
- **Deterministic Policy**: π(s) = a
- **Stochastic Policy**: π(a|s) = probability of taking action a in state s

### Value Functions

**State Value Function (V^π(s)):**
The expected cumulative reward starting from state s following policy π:
```
V^π(s) = E[Σ γ^t * r_{t+1} | s_0 = s, π]
```

**Action Value Function (Q^π(s, a)):**
The expected cumulative reward starting from state s, taking action a, then following policy π:
```
Q^π(s, a) = E[Σ γ^t * r_{t+1} | s_0 = s, a_0 = a, π]
```

### Bellman Equations

**Bellman Equation for V^π:**
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s, a) [R(s, a, s') + γ V^π(s')]
```

**Bellman Equation for Q^π:**
```
Q^π(s, a) = Σ_{s'} P(s'|s, a) [R(s, a, s') + γ Σ_{a'} π(a'|s') Q^π(s', a')]
```

### Optimal Policy

The optimal policy π* maximizes the value function:
```
π*(s) = argmax_a Q*(s, a)
```

Where Q* is the optimal action-value function:
```
Q*(s, a) = max_π Q^π(s, a)
```

## Deep Q-Network (DQN)

### Overview

DQN combines Q-learning with deep neural networks to handle high-dimensional state spaces. It was introduced by Mnih et al. in 2015.

### Key Components

1. **Neural Network Q-Function Approximator**
   - Input: State s
   - Output: Q-values for all actions Q(s, a) for all a ∈ A

2. **Experience Replay**
   - Stores experiences (s, a, r, s', done) in a replay buffer
   - Samples random batches for training
   - Breaks correlation between consecutive samples

3. **Target Network**
   - Separate network for computing target Q-values
   - Updates less frequently than main network
   - Provides stable targets for learning

### Algorithm

**DQN Algorithm:**

1. Initialize Q-network θ and target network θ' = θ
2. Initialize replay buffer D
3. For each episode:
   - Initialize state s
   - For each step:
     - Select action a using ε-greedy policy:
       ```
       a = {random action with probability ε
            argmax_a Q(s, a; θ) otherwise
       ```
     - Execute a, observe r, s'
     - Store (s, a, r, s', done) in D
     - Sample batch (s_i, a_i, r_i, s'_i, done_i) from D
     - Compute targets:
       ```
       y_i = {r_i if done_i
              r_i + γ max_a' Q(s'_i, a'; θ') otherwise
       ```
     - Update Q-network by minimizing:
       ```
       L = E[(y_i - Q(s_i, a_i; θ))²]
       ```
     - Every C steps: θ' ← θ
     - s ← s'

### Loss Function

The loss function is the mean squared error between predicted and target Q-values:

```
L(θ) = E_{(s,a,r,s') ~ D} [(r + γ max_{a'} Q(s', a'; θ') - Q(s, a; θ))²]
```

### Advantages

- Handles high-dimensional state spaces
- Learns from experience replay (efficient data usage)
- Stable learning with target network

### Limitations

- Overestimation bias in Q-values
- All experiences treated equally (no prioritization)
- Single value estimate (no separation of value and advantage)

## Double DQN

### Problem: Overestimation Bias

Standard DQN suffers from overestimation bias because:
```
E[max_a Q(s, a)] ≥ max_a E[Q(s, a)]
```

The max operation causes overestimation, especially early in training.

### Solution

Double DQN uses two networks:
- **Main Network (θ)**: Selects the best action
- **Target Network (θ')**: Evaluates that action

**Target Calculation:**
```
y = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ')
```

This decouples action selection from action evaluation, reducing overestimation.

### Algorithm Modification

The only change from standard DQN is in the target calculation:

**Standard DQN:**
```
y = r + γ max_{a'} Q(s', a'; θ')
```

**Double DQN:**
```
a* = argmax_{a'} Q(s', a'; θ)
y = r + γ Q(s', a*; θ')
```

### Benefits

- Reduces overestimation bias
- More stable learning
- Better final performance

## Dueling DQN

### Motivation

In many states, the value of being in that state is more important than the specific action taken. Dueling DQN separates these concerns.

### Architecture

**Standard DQN:**
```
State → Layers → Q(s, a)
```

**Dueling DQN:**
```
State → Shared Layers → Value Stream → V(s)
                      → Advantage Stream → A(s, a)
```

**Q-Value Calculation:**
```
Q(s, a) = V(s) + (A(s, a) - mean_{a'} A(s, a'))
```

The subtraction of the mean ensures that:
- The advantage has zero mean
- V(s) represents the value of being in state s
- A(s, a) represents the advantage of taking action a over average

### Mathematical Justification

The Q-function can be decomposed as:
```
Q(s, a) = V(s) + A(s, a)
```

However, this is unidentifiable (infinitely many solutions). The solution:
```
Q(s, a) = V(s) + (A(s, a) - max_{a'} A(s, a'))
```

Or more commonly:
```
Q(s, a) = V(s) + (A(s, a) - mean_{a'} A(s, a'))
```

This makes the representation identifiable.

### Benefits

- Better learning in states where action choice is less critical
- More efficient representation
- Better generalization

## Prioritized Experience Replay

### Motivation

Not all experiences are equally useful for learning. Experiences with high TD-error are more informative and should be sampled more frequently.

### Priority Assignment

Priority is based on TD-error:
```
priority_i = |δ_i| + ε
```

Where:
- δ_i = TD-error for experience i
- ε = small positive constant (prevents zero priority)

Priority is raised to power α:
```
P(i) = priority_i^α / Σ_j priority_j^α
```

Where:
- α = 0: uniform sampling (standard replay)
- α = 1: fully prioritized sampling

### Importance Sampling

To correct for the bias introduced by non-uniform sampling, importance sampling weights are used:

```
w_i = (N * P(i))^(-β)
```

Where:
- N = buffer size
- β = importance sampling exponent (0 = no correction, 1 = full correction)

β is annealed from β_0 to 1 during training.

### Implementation: Sum-Tree

For efficient sampling, a sum-tree data structure is used:
- O(log N) for sampling
- O(log N) for priority updates
- O(1) for finding maximum priority

### Algorithm

1. **Store Experience:**
   - Compute TD-error: δ = |r + γ max Q(s', a') - Q(s, a)|
   - Set priority: p = |δ|^α

2. **Sample Batch:**
   - Sample experiences proportional to priority
   - Compute importance sampling weights
   - Update β

3. **Update Priorities:**
   - After training, compute new TD-errors
   - Update priorities in sum-tree

### Benefits

- More efficient learning (focus on important experiences)
- Faster convergence
- Better final performance

## Neural Network Architectures

### Standard DQN Architecture

```
Input (State) → FC(128) → ReLU → FC(128) → ReLU → FC(64) → ReLU → Output (Q-values)
```

**Components:**
- Fully Connected (FC) layers
- ReLU activation
- Linear output layer

### Dueling DQN Architecture

```
Input (State) → Shared FC Layers → ┬─ Value Stream → V(s)
                                    └─ Advantage Stream → A(s, a)
                                    
                                    Q(s, a) = V(s) + (A(s, a) - mean A(s, a'))
```

**Components:**
- Shared feature extraction layers
- Separate value and advantage streams
- Combination layer

### Weight Initialization

Xavier (Glorot) Uniform Initialization:
```
W ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
```

This ensures:
- Gradients don't explode or vanish
- Faster convergence

### Activation Functions

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```

**Advantages:**
- Computationally efficient
- Reduces vanishing gradient problem
- Sparse activation

**ELU (Exponential Linear Unit):**
```
f(x) = {x if x > 0
        α(e^x - 1) if x ≤ 0
```

**Advantages:**
- Smooth gradients
- Better for deep networks

## Tools and Frameworks

### PyTorch

**Overview:**
PyTorch is a deep learning framework that provides:
- Dynamic computational graphs
- Automatic differentiation
- GPU acceleration
- Rich ecosystem of tools

**Key Features Used:**
- `torch.nn.Module`: Base class for neural networks
- `torch.optim`: Optimizers (Adam)
- `torch.nn.functional`: Activation functions, loss functions
- `torch.autograd`: Automatic differentiation

**Tensor Operations:**
- Efficient GPU computation
- Automatic gradient computation
- Memory-efficient operations

### Kivy

**Overview:**
Kivy is a Python framework for developing multi-touch applications.

**Key Features Used:**
- Widget-based UI
- Graphics rendering
- Event handling
- Clock scheduling

**Components:**
- `Widget`: Base UI component
- `Canvas`: Drawing surface
- `Properties`: Reactive properties
- `Clock`: Time-based scheduling

### NumPy

**Overview:**
NumPy provides efficient numerical computing.

**Key Features Used:**
- Multi-dimensional arrays
- Mathematical operations
- Efficient memory management
- Vectorized operations

### Matplotlib

**Overview:**
Matplotlib provides plotting and visualization.

**Key Features Used:**
- Line plots for training curves
- Figure management
- Export capabilities

### PyYAML

**Overview:**
PyYAML provides YAML parsing for configuration management.

**Key Features Used:**
- Configuration file parsing
- Nested data structures
- Type safety

### Testing Frameworks

**pytest:**
- Test discovery
- Fixtures for test setup
- Parameterized tests
- Coverage reporting

**pytest-cov:**
- Code coverage measurement
- HTML and terminal reports

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
3. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.
4. Schaul, T., et al. (2016). "Prioritized Experience Replay." ICLR.
5. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction." MIT Press.


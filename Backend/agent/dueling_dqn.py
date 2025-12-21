"""
Optimized Dueling DQN Agent for SAGIN Routing
PyTorch implementation với advanced optimizations và performance improvements
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from collections import deque, namedtuple
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

# Define experience structure for better performance
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'priority'])


class DuelingDQN(nn.Module):
    """
    Optimized Dueling DQN Network với advanced architecture
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = 'elu',
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # Advanced activation functions
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)  # Better for DQN
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            self.activation = nn.ELU(alpha=1.0)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(state_dim) if use_layer_norm else nn.Identity()
        
        # Shared feature layers với residual connections
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Normalization
            if use_layer_norm:
                self.shared_layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            self.shared_layers.append(self.activation)
            
            # Dropout for regularization
            if dropout_rate > 0:
                self.shared_layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        self.shared_output_dim = input_dim
        
        # Value stream (V(s)) - optimized
        self.value_stream = nn.Sequential(
            nn.Linear(self.shared_output_dim, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, 1)
        )
        
        # Advantage stream (A(s, a)) - optimized
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.shared_output_dim, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, action_dim)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Advanced weight initialization"""
        if isinstance(module, nn.Linear):
            # He initialization for ELU/ReLU
            nn.init.kaiming_normal_(module.weight, 
                                  mode='fan_in', 
                                  nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass với residual connections và normalization"""
        # Input normalization
        x = self.input_norm(state)
        
        # Shared layers với residual connections
        for i, layer in enumerate(self.shared_layers):
            if isinstance(layer, nn.Linear) and i > 0:
                # Potential residual connection
                residual = x
                x = layer(x)
                if x.shape == residual.shape:
                    x = x + residual  # Residual connection
            else:
                x = layer(x)
        
        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine với advanced aggregation
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value only - useful for debugging"""
        x = self.input_norm(state)
        
        for layer in self.shared_layers:
            x = layer(x)
        
        return self.value_stream(x)
    
    def get_advantage(self, state: torch.Tensor) -> torch.Tensor:
        """Get advantage values only - useful for debugging"""
        x = self.input_norm(state)
        
        for layer in self.shared_layers:
            x = layer(x)
        
        return self.advantage_stream(x)


class PrioritizedReplayBuffer:
    """
    Optimized Prioritized Experience Replay Buffer
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Cache for performance
        self._max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience với priority"""
        experience = Experience(state, action, reward, next_state, done, self._max_priority)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities[self.position] = self._max_priority
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self._max_priority
        
        self.position = (self.position + 1) % self.capacity
        self._max_priority = max(self._max_priority, self.priorities.max())
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch với priorities"""
        if self.size == 0:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] if self.size < self.capacity else self.priorities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(probabilities), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.BoolTensor(np.array([e.done for e in experiences]))
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        return self.size


class ReplayBuffer:
    """
    Optimized standard replay buffer với performance improvements
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._cache = None  # Cache for sampled data
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience - invalidate cache"""
        self.buffer.append((state, action, reward, next_state, done))
        self._cache = None  # Invalidate cache
    
    def sample(self, batch_size: int) -> Tuple:
        """Optimized sampling với caching"""
        if len(self.buffer) < batch_size:
            return None
        
        # Use cache if available và valid
        if self._cache is not None and len(self._cache[0]) == batch_size:
            return self._cache
        
        # Sample batch
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to tensors - optimized
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int64)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.bool_)
        
        # Convert to tensors
        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        rewards_t = torch.from_numpy(rewards)
        next_states_t = torch.from_numpy(next_states)
        dones_t = torch.from_numpy(dones)
        
        # Cache the result
        self._cache = (states_t, actions_t, rewards_t, next_states_t, dones_t)
        
        return self._cache
    
    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQNAgent:
    """
    Optimized Dueling DQN Agent với advanced features
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or {}
        dqn_config = self.config.get('rl_agent', {}).get('dqn', {})
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing DuelingDQN Agent on device: {self.device}")
        
        # Advanced hyperparameters
        self.lr = dqn_config.get('learning_rate', 0.0001)
        self.gamma = dqn_config.get('gamma', 0.99)
        self.epsilon_start = dqn_config.get('exploration_initial_eps', 1.0)
        self.epsilon_end = dqn_config.get('exploration_final_eps', 0.05)
        self.epsilon_decay = dqn_config.get('exploration_decay', 0.999)
        self.epsilon_decay_strategy = dqn_config.get('epsilon_decay_strategy', 'linear')  # 🔧 NEW: linear or exponential
        self.target_update_interval = dqn_config.get('target_update_interval', 100)  # 🔧 FIX: Default 100
        self.batch_size = dqn_config.get('batch_size', 32)  # 🔧 FIX: Reduced from 64
        self.buffer_size = dqn_config.get('buffer_size', 100000)
        self.learning_starts = dqn_config.get('learning_starts', 256)  # 🔧 FIX: Reduced from 5000
        
        # Advanced features
        self.use_double_dqn = dqn_config.get('use_double_dqn', True)
        self.use_prioritized_replay = dqn_config.get('use_prioritized_replay', False)
        self.gradient_clip = dqn_config.get('gradient_clip', 10.0)
        self.tau = dqn_config.get('tau', 0.005)  # 🔧 FIX: Default soft update
        
        # Network architecture
        dueling_config = dqn_config.get('dueling', {})
        hidden_dims = dueling_config.get('hidden_dims', [512, 256, 128])
        activation = dueling_config.get('activation_fn', 'elu')
        dropout_rate = dueling_config.get('dropout_rate', 0.1)
        use_layer_norm = dueling_config.get('use_layer_norm', True)
        
        # Create optimized networks
        self.q_network = DuelingDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        ).to(self.device)
        
        self.target_network = DuelingDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        ).to(self.device)
        
        # Initialize target network
        self._update_target_network(1.0)  # Hard update initially
        
        # Optimizer với learning rate scheduling
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=self.lr,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1000
        )
        
        # Replay buffer
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=self.buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)
        
        # Training state
        self.step_count = 0
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        self.episode_count = 0
        
        # Statistics
        self.training_losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)
        self.grad_norms = deque(maxlen=100)
        
        logger.info(f"DuelingDQN Agent initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> int:
        """
        Optimized action selection với advanced exploration
        """
        if not deterministic and random.random() < self.epsilon:
            # Exploratory action với action mask
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    return int(random.choice(valid_actions))
            return random.randint(0, self.action_dim - 1)
        
        # Greedy action với temperature
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Apply action mask và temperature
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values + (1 - mask_tensor) * -1e9
            
            # Apply temperature
            if temperature != 1.0 and not deterministic:
                q_values = q_values / temperature
                action_probs = F.softmax(q_values, dim=1)
                action = torch.multinomial(action_probs, 1).item()
            else:
                action = q_values.argmax().item()
            
            if deterministic:
                max_q = q_values.max().item()
                if max_q < -100:
                    logger.warning(
                        f"All Q-values are very low in deterministic mode: {max_q:.2f}. "
                        f"This may indicate the model needs more training or the state is problematic."
                    )
        
        return action
    
    def update_epsilon(self, total_steps: int, max_steps: int):
        """
        🔧 FIX: Support both linear and exponential epsilon decay
        Linear decay is more stable for learning
        """
        if self.epsilon_decay_strategy == 'linear':
            # Linear decay: epsilon decreases linearly from start to end
            progress = min(1.0, total_steps / max(max_steps, 1))
            self.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
            self.epsilon = max(self.epsilon_end, min(self.epsilon_start, self.epsilon))
        else:
            # Exponential decay (original)
            self.epsilon = max(self.epsilon_end, 
                              self.epsilon_start * (self.epsilon_decay ** total_steps))
    
    def train_step(self) -> Optional[Dict]:
        """
        Optimized training step với advanced features
        """
        if len(self.replay_buffer) < self.learning_starts:
            return None
        
        # Sample batch
        if self.use_prioritized_replay:
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones, weights, indices = batch
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones(self.batch_size, 1)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values - Double DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Use online network to select actions
                next_actions = self.q_network(next_states).argmax(1)
                # Use target network to evaluate actions
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.max(1)[0]
            
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Compute loss với Huber loss for stability
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss * weights.squeeze()).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping và monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 
            self.gradient_clip
        )
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            with torch.no_grad():
                priorities = torch.abs(current_q - target_q).detach().cpu().numpy() + 1e-6
                self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self._update_target_network(self.tau)
        
        # Update learning rate
        self.lr_scheduler.step(loss.detach())  # Fix: detach to avoid gradient warning
        
        # Collect statistics
        self.training_losses.append(loss.item())
        self.q_values.append(current_q.mean().item())
        self.grad_norms.append(grad_norm.item())
        
        metrics = {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'target_q_value': target_q.mean().item(),
            'grad_norm': grad_norm.item(),
            'epsilon': self.epsilon,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def _update_target_network(self, tau: float = 1.0):
        """Update target network với soft/hard updates"""
        if tau == 1.0:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update
            for target_param, param in zip(self.target_network.parameters(), 
                                         self.q_network.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )
    
    def update_target_network(self, tau: float = 1.0):
        """Public method to update target network"""
        self._update_target_network(tau)
    
    def save(self, filepath: str):
        """Save agent với optimizations"""
        # Create directory if not exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'training_losses': list(self.training_losses),
            'q_values': list(self.q_values),
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent với error handling"""
        try:
            # PyTorch 2.6+ changed default weights_only from False to True
            # Model được lưu từ training nên trust source, use weights_only=False
            try:
                # Try với weights_only=False first (for PyTorch 2.6+)
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            except (TypeError, Exception) as e:
                # PyTorch < 2.6 không có weights_only parameter, hoặc có lỗi khác
                # Try without weights_only parameter
                try:
                    checkpoint = torch.load(filepath, map_location=self.device)
                except Exception as e2:
                    # Final fallback: force weights_only=False explicitly
                    import pickle
                    checkpoint = torch.load(filepath, map_location=self.device, weights_only=False, pickle_module=pickle)
            
            # Check if action_dim matches
            checkpoint_action_dim = None
            if 'action_dim' in checkpoint:
                checkpoint_action_dim = checkpoint['action_dim']
            elif 'q_network_state_dict' in checkpoint:
                # Try to infer from checkpoint
                for key in checkpoint['q_network_state_dict'].keys():
                    if 'advantage_stream.6.weight' in key:
                        checkpoint_action_dim = checkpoint['q_network_state_dict'][key].shape[0]
                        break
            
            if checkpoint_action_dim and checkpoint_action_dim != self.action_dim:
                logger.warning(
                    f"Action dimension mismatch: checkpoint has {checkpoint_action_dim}, "
                    f"current model has {self.action_dim}. Loading with strict=False (output layer will not be loaded)."
                )
                # Load with strict=False to skip mismatched layers
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'], strict=False)
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'], strict=False)
            else:
                # Normal load
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            
            # Load optimizer (may fail if architecture changed, that's ok)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}. Using fresh optimizer.")
            
            if 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            self.step_count = checkpoint.get('step_count', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            
            logger.info(f"Agent loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading agent from {filepath}: {e}")
            raise
    
    def get_state_dict(self):
        """Get state dict for caching"""
        return self.q_network.state_dict()
    
    def get_state(self) -> Dict:
        """Get training state for resuming"""
        return {
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'training_losses': list(self.training_losses),
            'q_values': list(self.q_values),
            'grad_norms': list(self.grad_norms),
        }
    
    def eval(self):
        """Set to evaluation mode"""
        self.q_network.eval()
        self.target_network.eval()
    
    def train_mode(self):
        """Set to training mode"""
        self.q_network.train()
        self.target_network.eval()  # Target network always in eval mode
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'buffer_size': len(self.replay_buffer),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'avg_loss': np.mean(self.training_losses) if self.training_losses else 0,
            'avg_q_value': np.mean(self.q_values) if self.q_values else 0,
            'avg_grad_norm': np.mean(self.grad_norms) if self.grad_norms else 0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
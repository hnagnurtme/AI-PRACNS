"""
Base abstract agent class for reinforcement learning agents.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.

    This class defines the interface that all agents must implement.
    """

    def __init__(self, action_space_size: int, observation_space_size: int):
        """
        Initialize base agent.

        Args:
            action_space_size: Size of the action space
            observation_space_size: Size of the observation space
        """
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.training = True

    @abstractmethod
    def select_action(self, state: np.ndarray, num_valid_actions: Optional[int] = None,
                     is_training: bool = True) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current state observation
            num_valid_actions: Number of valid actions (if action space is variable)
            is_training: Whether the agent is in training mode

        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """
        Update agent's knowledge based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: str):
        """
        Save agent checkpoint to file.

        Args:
            filepath: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: str):
        """
        Load agent checkpoint from file.

        Args:
            filepath: Path to load checkpoint from
        """
        pass

    def train(self):
        """Set agent to training mode"""
        self.training = True

    def eval(self):
        """Set agent to evaluation mode"""
        self.training = False

    def get_config(self) -> dict:
        """
        Get agent configuration.

        Returns:
            Dictionary with agent configuration
        """
        return {
            'action_space_size': self.action_space_size,
            'observation_space_size': self.observation_space_size,
            'training': self.training
        }


class RandomAgent(BaseAgent):
    """
    Random agent that selects actions uniformly at random.
    Useful as a baseline for comparison.
    """

    def __init__(self, action_space_size: int, observation_space_size: int):
        super().__init__(action_space_size, observation_space_size)

    def select_action(self, state: np.ndarray, num_valid_actions: Optional[int] = None,
                     is_training: bool = True) -> int:
        """Select random action"""
        max_actions = num_valid_actions if num_valid_actions else self.action_space_size
        return np.random.randint(0, max_actions)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """Random agent doesn't learn"""
        pass

    def save_checkpoint(self, filepath: str):
        """Random agent has no state to save"""
        pass

    def load_checkpoint(self, filepath: str):
        """Random agent has no state to load"""
        pass

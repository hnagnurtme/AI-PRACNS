# training/trainer.py
from agents.DqnAgent import DqnAgent
from agents.InMemoryReplayBuffer import InMemoryReplayBuffer
from typing import Tuple

def train_agent_batch(agent: DqnAgent, buffer: InMemoryReplayBuffer, batch_size: int, target_update_freq: int) -> Tuple[float, float, int]:
    """Thực hiện một bước huấn luyện và quản lý cập nhật Target."""
    
    if buffer.get_size() < batch_size * 2:
        return 0.0, agent.epsilon, buffer.get_size()

    experiences = buffer.sample(batch_size)
    loss = agent.learn(experiences)
    agent.decay_epsilon()
    
    if agent._learn_step_counter % target_update_freq == 0:
        agent.update_target_net()
        
    return loss, agent.epsilon, buffer.get_size()
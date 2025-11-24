import torch
import random
import math
import numpy as np

from .dqn_model import OUTPUT_SIZE

def get_epsilon(steps_done: int, epsilon_params: dict) -> float:
    eps_start = epsilon_params['start']
    eps_end = epsilon_params['end']
    eps_decay = epsilon_params['decay']
    return eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)

def select_action(q_network: torch.nn.Module, 
                  state_vector: np.ndarray, 
                  steps_done: int, 
                  num_valid_actions: int, 
                  device: torch.device,
                  epsilon_params: dict) -> int:
    epsilon = get_epsilon(steps_done, epsilon_params)
    
    if random.random() < epsilon:
        if num_valid_actions <= 0:
            return 0
        return random.randint(0, num_valid_actions - 1)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            
            mask = torch.full_like(q_values, float('-inf'))
            safe_actions = min(num_valid_actions, OUTPUT_SIZE)
            mask[:, :safe_actions] = 0
            
            masked_q_values = q_values + mask
            return masked_q_values.argmax(dim=1).item()
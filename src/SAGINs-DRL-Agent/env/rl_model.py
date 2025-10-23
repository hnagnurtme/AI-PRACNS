import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    def __init__(self, state_size, action_size, checkpoint_path='rl_checkpoint.pth'):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=2000)
        
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint()
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())
    
    def find_path(self, env, max_steps=30):
        path = [env.current_node]
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = self.act(state)
            # Get all possible next nodes based on connectivity
            next_nodes = [n for n in env.node_ids if n not in path and env._is_connected(path[-1], n)]
            if not next_nodes:
                break
            # No priority, let RL decide based on Q-values
            next_node = next_nodes[action % len(next_nodes)]  # Map action to valid next node
            next_state, reward, done = env.step([next_node])
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
            path.append(next_node)
            steps += 1
        if done and env.current_node == env.dest_node:
            return path
        return path if path[-1] == env.dest_node else None
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        q_eval = self.model(states).gather(1, actions).squeeze()
        q_next = self.model(next_states).max(1)[0]
        q_target = rewards + (1 - dones) * self.gamma * q_next
        
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.checkpoint_path)
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print("Loaded RL checkpoint")
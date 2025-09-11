"""
Proximal Policy Optimization (PPO) Agent for Portfolio Optimization
Integrated with LSTM predictions and risk management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import gymnasium as gym
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 dropout: float = 0.1):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        feature_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*feature_layers)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim),
            nn.Softmax(dim=-1)  # Portfolio weights should sum to <= 1
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        features = self.shared_features(state)
        
        # Actor output (action probabilities/weights)
        action_probs = self.actor(features)
        
        # Critic output (state value)
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        """Get action from the policy"""
        action_probs, value = self.forward(state)
        
        if deterministic:
            # Use most probable action for evaluation
            action = action_probs
        else:
            # Sample from the distribution for training
            # Add small noise for exploration
            noise = torch.randn_like(action_probs) * 0.01
            action = action_probs + noise
            action = torch.clamp(action, 0, 1)
            
            # Normalize to ensure sum <= 1
            action = action / (torch.sum(action, dim=-1, keepdim=True) + 1e-8)
        
        return action, value
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        action_probs, values = self.forward(states)
        
        # Calculate action log probabilities
        # Using Dirichlet distribution for portfolio weights
        alpha = action_probs * 10 + 1e-8  # Convert to Dirichlet parameters
        dist = torch.distributions.Dirichlet(alpha)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class PPOAgent:
    """
    Proximal Policy Optimization Agent for Portfolio Management
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 update_epochs: int = 10,
                 mini_batch_size: int = 64,
                 buffer_size: int = 2048):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.buffer_size = buffer_size
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': lr_actor},
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Experience buffer
        self.buffer = PPOBuffer(buffer_size, state_dim, action_dim, self.device)
        
        # Training statistics
        self.total_steps = 0
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100)
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Get action from the policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.actor_critic.get_action(state_tensor, deterministic)
        
        action_np = action.cpu().numpy().squeeze()
        value_np = value.cpu().numpy().squeeze()
        
        return action_np, value_np
    
    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        """Store experience in buffer"""
        self.buffer.store(state, action, reward, next_state, done, value, log_prob)
    
    def update(self) -> Dict[str, float]:
        """Update the policy using PPO"""
        if len(self.buffer) < self.buffer_size:
            return {}
        
        # Get batch data
        states, actions, rewards, next_states, dones, old_values, old_log_probs = self.buffer.get()
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        update_stats = []
        
        for epoch in range(self.update_epochs):
            # Create mini-batches
            batch_indices = np.arange(len(states))
            np.random.shuffle(batch_indices)
            
            for start_idx in range(0, len(states), self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                batch_idx = batch_indices[start_idx:end_idx]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                
                # Evaluate current policy
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calculate policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (actor_loss + 
                             self.value_coef * value_loss + 
                             self.entropy_coef * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Store statistics
                update_stats.append({
                    'actor_loss': actor_loss.item(),
                    'critic_loss': value_loss.item(),
                    'entropy': entropy.mean().item(),
                    'kl_divergence': (batch_old_log_probs - new_log_probs).mean().item()
                })
        
        # Update learning rate
        self.scheduler.step()
        
        # Clear buffer
        self.buffer.clear()
        
        # Calculate average statistics
        avg_stats = {}
        if update_stats:
            for key in update_stats[0].keys():
                avg_value = np.mean([stat[key] for stat in update_stats])
                avg_stats[key] = avg_value
                self.training_stats[key].append(avg_value)
        
        return avg_stats
    
    def _calculate_gae(self, rewards, values, dones):
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards) - 1)):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
        
        returns = advantages + values[:-1]  # Remove last value
        
        return advantages, returns
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'eps_clip': self.eps_clip,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            },
            'training_stats': dict(self.training_stats),
            'total_steps': self.total_steps
        }
        
        torch.save(checkpoint, filepath)
        print(f"PPO model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        
        print(f"PPO model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return {key: list(values) for key, values in self.training_stats.items()}


class PPOBuffer:
    """
    Experience buffer for PPO
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        
        # Buffers
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, state, action, reward, next_state, done, value, log_prob):
        """Store experience"""
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.ptr] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[self.ptr] = torch.FloatTensor(next_state).to(self.device)
        self.dones[self.ptr] = torch.FloatTensor([done]).to(self.device)
        
        # Handle value conversion more robustly
        if torch.is_tensor(value):
            self.values[self.ptr] = value.clone().detach().to(self.device)
        elif isinstance(value, np.ndarray):
            self.values[self.ptr] = torch.FloatTensor([value.item()]).to(self.device)
        else:
            self.values[self.ptr] = torch.FloatTensor([value]).to(self.device)
        
        # Handle log_prob conversion more robustly
        if torch.is_tensor(log_prob):
            self.log_probs[self.ptr] = log_prob.clone().detach().to(self.device)
        elif isinstance(log_prob, np.ndarray):
            self.log_probs[self.ptr] = torch.FloatTensor([log_prob.item()]).to(self.device)
        else:
            self.log_probs[self.ptr] = torch.FloatTensor([log_prob]).to(self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self):
        """Get all stored experiences"""
        return (self.states[:self.size], 
                self.actions[:self.size],
                self.rewards[:self.size],
                self.next_states[:self.size],
                self.dones[:self.size],
                self.values[:self.size],
                self.log_probs[:self.size])
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size


class PortfolioPPOTrainer:
    """
    High-level trainer for PPO agent on portfolio optimization
    """
    
    def __init__(self, 
                 env: gym.Env,
                 agent: PPOAgent,
                 max_episodes: int = 1000,
                 max_steps_per_episode: int = 500,
                 eval_frequency: int = 50,
                 save_frequency: int = 100):
        
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.evaluation_rewards = []
        
    def train(self, save_path: str = None) -> Dict[str, List]:
        """Train the PPO agent"""
        print(f"Starting PPO training for {self.max_episodes} episodes...")
        
        pbar = tqdm(range(self.max_episodes), desc="PPO Training", unit="episode")
        for episode in pbar:
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                # Get action
                action, value = self.agent.get_action(state)
                
                # Take environment step
                next_state, reward, done, info = self.env.step(action)
                
                # Calculate log probability (simplified for portfolio weights)
                log_prob = -np.sum(action * np.log(action + 1e-8))
                
                # Store experience
                self.agent.store_transition(
                    state, action, reward, next_state, done, value, log_prob
                )
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Update agent
            update_stats = self.agent.update()
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Update tqdm with current stats
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                pbar.set_postfix({
                    'Avg Reward': f'{avg_reward:.4f}',
                    'Episode Reward': f'{episode_reward:.4f}',
                    'Steps': episode_length
                })
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_reward = self.evaluate_agent()
                self.evaluation_rewards.append(eval_reward)
                pbar.write(f"🎯 Evaluation at episode {episode}: Reward = {eval_reward:.4f}")
            
            # Save model
            if save_path and episode % self.save_frequency == 0 and episode > 0:
                model_path = f"{save_path}_episode_{episode}.pth"
                self.agent.save_model(model_path)
        
        # Final save
        if save_path:
            self.agent.save_model(f"{save_path}_final.pth")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_rewards': self.evaluation_rewards
        }
    
    def evaluate_agent(self, num_episodes: int = 5) -> float:
        """Evaluate the agent's performance"""
        total_reward = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.max_steps_per_episode):
                action, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
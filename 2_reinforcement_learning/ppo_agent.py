"""
PPO (Proximal Policy Optimization) Agent
Learns to improve through self-play and rewards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple
import numpy as np

import sys
sys.path.append('..')
from config import RL, DEVICE


class PPOAgent(nn.Module):
    """
    PPO Agent with actor-critic architecture
    Actor: Decides what actions to take
    Critic: Evaluates how good the current state is
    """
    
    def __init__(
        self,
        obs_dim: int = 512,
        action_dim: int = 573,
        spatial_dim: int = 84 * 84
    ):
        super().__init__()
        
        # Feature extractor (can load from imitation model)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, RL['policy_hidden'][0]),
            nn.ReLU(),
            nn.Linear(RL['policy_hidden'][0], RL['policy_hidden'][1]),
            nn.ReLU(),
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=RL['policy_hidden'][1],
            hidden_size=RL['lstm_hidden'],
            num_layers=RL['lstm_layers'],
            batch_first=True
        )
        
        # Policy network (actor) - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(RL['lstm_hidden'], RL['policy_hidden'][2]),
            nn.ReLU(),
            nn.Linear(RL['policy_hidden'][2], action_dim)
        )
        
        # Spatial policy (where to click)
        self.spatial_policy = nn.Sequential(
            nn.Linear(RL['lstm_hidden'], 256),
            nn.ReLU(),
            nn.Linear(256, spatial_dim)
        )
        
        # Value network (critic) - estimates state value
        self.value_head = nn.Sequential(
            nn.Linear(RL['lstm_hidden'], RL['value_hidden'][0]),
            nn.ReLU(),
            nn.Linear(RL['value_hidden'][0], RL['value_hidden'][1]),
            nn.ReLU(),
            nn.Linear(RL['value_hidden'][1], 1)
        )
        
        self.to(DEVICE)
        
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass through policy and value networks
        
        Args:
            obs: Observations [batch, seq_len, obs_dim]
            hidden: LSTM hidden state
            
        Returns:
            action_logits: Action type logits
            spatial_logits: Spatial action logits
            value: State value estimates
            hidden: Updated LSTM hidden state
        """
        # Extract features
        features = self.feature_extractor(obs)
        
        # Process through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(features)
        else:
            lstm_out, hidden = self.lstm(features, hidden)
        
        # Get policy outputs
        action_logits = self.policy_head(lstm_out)
        spatial_logits = self.spatial_policy(lstm_out)
        
        # Get value estimate
        value = self.value_head(lstm_out)
        
        return action_logits, spatial_logits, value, hidden
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
        action: torch.Tensor = None,
        spatial_action: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get action, log probability, and value
        Used during rollout collection
        
        Args:
            obs: Observation
            hidden: LSTM hidden state
            action: If provided, compute log prob of this action
            spatial_action: If provided, compute log prob of this spatial action
            
        Returns:
            Dictionary with actions, log probs, values, entropy
        """
        action_logits, spatial_logits, value, new_hidden = self.forward(obs, hidden)
        
        # Create categorical distributions
        action_dist = Categorical(logits=action_logits)
        spatial_dist = Categorical(logits=spatial_logits.view(*spatial_logits.shape[:-1], -1))
        
        # Sample actions if not provided
        if action is None:
            action = action_dist.sample()
        if spatial_action is None:
            spatial_action = spatial_dist.sample()
        
        # Compute log probabilities
        action_log_prob = action_dist.log_prob(action)
        spatial_log_prob = spatial_dist.log_prob(spatial_action)
        
        # Compute entropy (for exploration bonus)
        action_entropy = action_dist.entropy()
        spatial_entropy = spatial_dist.entropy()
        
        return {
            'action': action,
            'spatial_action': spatial_action,
            'action_log_prob': action_log_prob,
            'spatial_log_prob': spatial_log_prob,
            'value': value.squeeze(-1),
            'entropy': action_entropy + spatial_entropy,
            'hidden': new_hidden
        }
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        spatial_actions: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions (for PPO update)
        
        Args:
            obs: Observations
            actions: Actions taken
            spatial_actions: Spatial actions taken
            hidden: LSTM hidden state
            
        Returns:
            Dictionary with log probs, values, entropy
        """
        action_logits, spatial_logits, value, _ = self.forward(obs, hidden)
        
        # Create distributions
        action_dist = Categorical(logits=action_logits)
        spatial_dist = Categorical(logits=spatial_logits.view(*spatial_logits.shape[:-1], -1))
        
        # Compute log probabilities
        action_log_prob = action_dist.log_prob(actions)
        spatial_log_prob = spatial_dist.log_prob(spatial_actions)
        
        # Compute entropy
        action_entropy = action_dist.entropy()
        spatial_entropy = spatial_dist.entropy()
        
        return {
            'action_log_prob': action_log_prob,
            'spatial_log_prob': spatial_log_prob,
            'value': value.squeeze(-1),
            'entropy': action_entropy + spatial_entropy
        }


class PPOTrainer:
    """Handles PPO training updates"""
    
    def __init__(self, agent: PPOAgent, learning_rate: float = None):
        self.agent = agent
        
        lr = learning_rate or RL['learning_rate']
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        
        self.clip_epsilon = RL['clip_epsilon']
        self.vf_coef = RL['vf_coef']
        self.entropy_coef = RL['entropy_coef']
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        gamma: float = None,
        gae_lambda: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        This gives us a good estimate of how much better an action was
        compared to the average action in that state
        """
        gamma = gamma or RL['gamma']
        gae_lambda = gae_lambda or RL['gae_lambda']
        
        advantages = []
        gae = 0
        
        # Compute advantages backwards through time
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(DEVICE)
        
        return advantages, returns
    
    def update(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        spatial_action_batch: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_spatial_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        PPO update step
        
        Args:
            obs_batch: Observations
            action_batch: Actions taken
            spatial_action_batch: Spatial actions taken
            old_log_probs: Old action log probabilities
            old_spatial_log_probs: Old spatial action log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            
        Returns:
            Dictionary with loss statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Evaluate current policy
        eval_results = self.agent.evaluate_actions(
            obs_batch,
            action_batch,
            spatial_action_batch
        )
        
        # Compute ratios for PPO
        log_probs = eval_results['action_log_prob'] + eval_results['spatial_log_prob']
        old_log_probs_total = old_log_probs + old_spatial_log_probs
        
        ratios = torch.exp(log_probs - old_log_probs_total)
        
        # PPO clipped objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(eval_results['value'], returns)
        
        # Entropy bonus (encourages exploration)
        entropy = eval_results['entropy'].mean()
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': ((ratios - 1) - log_probs + old_log_probs_total).mean().item()
        }


def create_ppo_agent():
    """Factory function to create PPO agent"""
    agent = PPOAgent()
    
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Created PPO Agent:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {DEVICE}")
    
    return agent


if __name__ == "__main__":
    # Test PPO agent
    print("Testing PPO Agent...")
    
    agent = create_ppo_agent()
    trainer = PPOTrainer(agent)
    
    # Create dummy data
    batch_size = 32
    seq_len = 10
    obs_dim = 512
    
    obs = torch.randn(batch_size, seq_len, obs_dim).to(DEVICE)
    
    # Test forward pass
    outputs = agent.get_action_and_value(obs)
    
    print(f"\nOutput shapes:")
    print(f"  Action: {outputs['action'].shape}")
    print(f"  Spatial action: {outputs['spatial_action'].shape}")
    print(f"  Value: {outputs['value'].shape}")
    print(f"  Entropy: {outputs['entropy'].shape}")
    
    # Test GAE computation
    rewards = [0.1, 0.2, 0.3, 1.0]
    values = [0.5, 0.6, 0.7, 0.8]
    dones = [False, False, False, True]
    
    advantages, returns = trainer.compute_gae(rewards, values, dones)
    print(f"\nGAE test:")
    print(f"  Advantages: {advantages}")
    print(f"  Returns: {returns}")
    
    print("\n✓ PPO Agent test passed!")

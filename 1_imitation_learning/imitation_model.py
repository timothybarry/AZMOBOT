"""
Imitation Learning Model
Neural network that learns to mimic expert players from replays
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

import sys
sys.path.append('..')
from config import IMITATION, SPATIAL_ENCODER, ATTENTION, DEVICE


class SpatialEncoder(nn.Module):
    """Encodes screen and minimap observations using ResNet-style architecture"""
    
    def __init__(self, input_channels: int = 27):
        super().__init__()
        
        # ResNet-style blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Residual connection
        self.residual = nn.Conv2d(32, 128, kernel_size=1, stride=2)
        
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # Add residual connection
        residual = self.residual(x1)
        x3 = x3 + residual
        
        return x3


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for focusing on important spatial locations"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)  # Residual connection


class ImitationModel(nn.Module):
    """
    Main imitation learning model
    Learns to predict expert actions from game observations
    """
    
    def __init__(
        self,
        screen_channels: int = 27,
        minimap_channels: int = 7,
        num_actions: int = 573,
    ):
        super().__init__()
        
        # Spatial encoders
        self.screen_encoder = SpatialEncoder(screen_channels)
        self.minimap_encoder = SpatialEncoder(minimap_channels)
        
        # Calculate spatial feature dimensions after encoding
        # Starting from 84x84, after two stride-2 convs: 21x21
        self.spatial_dim = 21 * 21 * 128
        
        # Flatten and project spatial features
        self.screen_projection = nn.Linear(self.spatial_dim, IMITATION['embedding_dim'])
        self.minimap_projection = nn.Linear(self.spatial_dim, IMITATION['embedding_dim'])
        
        # Non-spatial features (game info like resources, supply, etc.)
        self.nonspatial_embed = nn.Linear(64, IMITATION['embedding_dim'])
        
        # Combine all features
        self.feature_fusion = nn.Linear(IMITATION['embedding_dim'] * 3, IMITATION['embedding_dim'])
        
        # LSTM for temporal processing (remembers game history)
        self.lstm = nn.LSTM(
            input_size=IMITATION['embedding_dim'],
            hidden_size=IMITATION['lstm_hidden'],
            num_layers=IMITATION['lstm_layers'],
            batch_first=True,
            dropout=IMITATION['dropout'] if IMITATION['lstm_layers'] > 1 else 0
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            IMITATION['lstm_hidden'],
            IMITATION['attention_heads']
        )
        
        # Action prediction heads
        self.action_type_head = nn.Linear(IMITATION['lstm_hidden'], num_actions)
        
        # Spatial action heads (where to click)
        self.spatial_head = nn.Linear(IMITATION['lstm_hidden'], 84 * 84)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(IMITATION['dropout'])
        
    def forward(
        self,
        screen: torch.Tensor,
        minimap: torch.Tensor,
        nonspatial: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            screen: [batch, seq_len, channels, height, width]
            minimap: [batch, seq_len, channels, height, width]
            nonspatial: [batch, seq_len, features]
            hidden: LSTM hidden state
            
        Returns:
            Dictionary with action predictions
        """
        batch_size, seq_len = screen.shape[:2]
        
        # Reshape for spatial encoding: [batch*seq_len, channels, H, W]
        screen_flat = screen.view(-1, *screen.shape[2:])
        minimap_flat = minimap.view(-1, *minimap.shape[2:])
        
        # Encode spatial features
        screen_features = self.screen_encoder(screen_flat)
        minimap_features = self.minimap_encoder(minimap_flat)
        
        # Flatten spatial dimensions
        screen_features = screen_features.view(batch_size * seq_len, -1)
        minimap_features = minimap_features.view(batch_size * seq_len, -1)
        
        # Project to embedding dimension
        screen_embed = self.screen_projection(screen_features)
        minimap_embed = self.minimap_projection(minimap_features)
        
        # Process non-spatial features
        nonspatial_flat = nonspatial.view(batch_size * seq_len, -1)
        nonspatial_embed = self.nonspatial_embed(nonspatial_flat)
        
        # Fuse all features
        combined = torch.cat([screen_embed, minimap_embed, nonspatial_embed], dim=-1)
        combined = self.feature_fusion(combined)
        combined = F.relu(combined)
        combined = self.dropout(combined)
        
        # Reshape back to sequences: [batch, seq_len, embedding]
        combined = combined.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(combined)
        else:
            lstm_out, hidden = self.lstm(combined, hidden)
        
        # Apply attention
        attended = self.attention(lstm_out)
        
        # Predict actions
        action_logits = self.action_type_head(attended)
        spatial_logits = self.spatial_head(attended)
        
        # Reshape spatial logits to [batch, seq_len, 84, 84]
        spatial_logits = spatial_logits.view(batch_size, seq_len, 84, 84)
        
        return {
            'action_type': action_logits,
            'spatial': spatial_logits,
            'hidden': hidden,
            'features': attended  # For visualization/analysis
        }
    
    def get_action(
        self,
        screen: torch.Tensor,
        minimap: torch.Tensor,
        nonspatial: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action for a single timestep (for deployment)
        
        Args:
            screen: [1, channels, height, width]
            minimap: [1, channels, height, width]  
            nonspatial: [1, features]
            hidden: LSTM hidden state
            deterministic: If True, take argmax; if False, sample
            
        Returns:
            action: Dictionary with action_type and spatial coordinates
            hidden: Updated hidden state
        """
        # Add sequence dimension
        screen = screen.unsqueeze(1)  # [1, 1, C, H, W]
        minimap = minimap.unsqueeze(1)
        nonspatial = nonspatial.unsqueeze(1)
        
        with torch.no_grad():
            outputs = self.forward(screen, minimap, nonspatial, hidden)
            
            # Get action type
            action_type_logits = outputs['action_type'].squeeze(1)  # [1, num_actions]
            
            if deterministic:
                action_type = torch.argmax(action_type_logits, dim=-1).item()
            else:
                action_type_probs = F.softmax(action_type_logits, dim=-1)
                action_type = torch.multinomial(action_type_probs, 1).item()
            
            # Get spatial location
            spatial_logits = outputs['spatial'].squeeze(1)  # [1, 84, 84]
            spatial_flat = spatial_logits.view(-1)
            
            if deterministic:
                spatial_idx = torch.argmax(spatial_flat).item()
            else:
                spatial_probs = F.softmax(spatial_flat, dim=-1)
                spatial_idx = torch.multinomial(spatial_probs, 1).item()
            
            # Convert flat index to 2D coordinates
            spatial_y = spatial_idx // 84
            spatial_x = spatial_idx % 84
        
        action = {
            'action_type': action_type,
            'x': spatial_x,
            'y': spatial_y
        }
        
        return action, outputs['hidden']


def create_imitation_model():
    """Factory function to create imitation model with config"""
    model = ImitationModel()
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Created Imitation Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {DEVICE}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Imitation Model...")
    
    model = create_imitation_model()
    
    # Create dummy inputs
    batch_size = 4
    seq_len = 10
    
    screen = torch.randn(batch_size, seq_len, 27, 84, 84).to(DEVICE)
    minimap = torch.randn(batch_size, seq_len, 7, 64, 64).to(DEVICE)
    # Upsample minimap to match screen size
    minimap = F.interpolate(
        minimap.view(-1, 7, 64, 64),
        size=(84, 84),
        mode='bilinear'
    ).view(batch_size, seq_len, 7, 84, 84)
    
    nonspatial = torch.randn(batch_size, seq_len, 64).to(DEVICE)
    
    # Forward pass
    outputs = model(screen, minimap, nonspatial)
    
    print(f"\nOutput shapes:")
    print(f"  Action type logits: {outputs['action_type'].shape}")
    print(f"  Spatial logits: {outputs['spatial'].shape}")
    print(f"  Features: {outputs['features'].shape}")
    
    # Test single-step action
    print("\nTesting single-step action prediction...")
    screen_single = torch.randn(1, 27, 84, 84).to(DEVICE)
    minimap_single = F.interpolate(
        torch.randn(1, 7, 64, 64).to(DEVICE),
        size=(84, 84),
        mode='bilinear'
    )
    nonspatial_single = torch.randn(1, 64).to(DEVICE)
    
    action, hidden = model.get_action(screen_single, minimap_single, nonspatial_single)
    
    print(f"Predicted action: {action}")
    print("\n✓ Model test passed!")

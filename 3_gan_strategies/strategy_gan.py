"""
GAN for Build Order Generation
Creates diverse, realistic strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple

import sys
sys.path.append('..')
from config import GAN, DEVICE


class BuildOrderGenerator(nn.Module):
    """
    Generator network that creates build orders from random noise
    
    Input: Random noise vector
    Output: Sequence of actions (build order)
    """
    
    def __init__(
        self,
        latent_dim: int = None,
        hidden_dims: list = None,
        output_length: int = None,
        vocab_size: int = None
    ):
        super().__init__()
        
        latent_dim = latent_dim or GAN['latent_dim']
        hidden_dims = hidden_dims or GAN['generator_hidden']
        self.output_length = output_length or GAN['build_order_length']
        self.vocab_size = vocab_size or GAN['action_vocab_size']
        
        # Initial projection
        self.fc_initial = nn.Linear(latent_dim, hidden_dims[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
            ])
        self.hidden = nn.Sequential(*layers)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # Output projection to vocabulary
        self.fc_output = nn.Linear(256, self.vocab_size)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate build order from noise
        
        Args:
            z: Noise vector [batch, latent_dim]
            
        Returns:
            Build order [batch, seq_len, vocab_size]
        """
        batch_size = z.shape[0]
        
        # Project noise
        x = F.leaky_relu(self.fc_initial(z), 0.2)
        
        # Hidden layers
        x = self.hidden(x)
        
        # Expand for sequence generation
        x = x.unsqueeze(1).repeat(1, self.output_length, 1)
        
        # Generate sequence
        lstm_out, _ = self.lstm(x)
        
        # Project to vocabulary
        output = self.fc_output(lstm_out)
        
        return output  # [batch, seq_len, vocab_size]
    
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """Generate build orders"""
        with torch.no_grad():
            z = torch.randn(num_samples, GAN['latent_dim']).to(DEVICE)
            return self.forward(z)


class BuildOrderDiscriminator(nn.Module):
    """
    Discriminator network that judges if a build order is real or fake
    
    Input: Build order sequence
    Output: Probability that it's real (from expert replays)
    """
    
    def __init__(
        self,
        hidden_dims: list = None,
        input_length: int = None,
        vocab_size: int = None
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or GAN['discriminator_hidden']
        self.input_length = input_length or GAN['build_order_length']
        self.vocab_size = vocab_size or GAN['action_vocab_size']
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, 128)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(512, 1)
        
        # Classification layers
        layers = []
        input_dim = 512  # Bidirectional LSTM output
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Judge if build order is real or fake
        
        Args:
            x: Build order [batch, seq_len] (indices) or [batch, seq_len, vocab_size] (logits)
            
        Returns:
            Scores [batch, 1]
        """
        # Convert logits to indices if needed
        if x.dim() == 3:
            x = torch.argmax(x, dim=-1)
        
        # Embed
        embedded = self.embedding(x)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention pooling
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classify
        score = self.classifier(attended)
        
        return score


class WGAN_GP:
    """
    Wasserstein GAN with Gradient Penalty
    More stable training than vanilla GAN
    """
    
    def __init__(
        self,
        generator: BuildOrderGenerator,
        discriminator: BuildOrderDiscriminator,
        lr_g: float = None,
        lr_d: float = None
    ):
        self.generator = generator.to(DEVICE)
        self.discriminator = discriminator.to(DEVICE)
        
        lr_g = lr_g or GAN['learning_rate_g']
        lr_d = lr_d or GAN['learning_rate_d']
        
        self.opt_g = optim.Adam(
            generator.parameters(),
            lr=lr_g,
            betas=(GAN['beta1'], GAN['beta2'])
        )
        
        self.opt_d = optim.Adam(
            discriminator.parameters(),
            lr=lr_d,
            betas=(GAN['beta1'], GAN['beta2'])
        )
        
        self.gp_lambda = GAN['gp_lambda']
        
    def gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP
        Encourages discriminator to have gradients with norm close to 1
        """
        batch_size = real_data.shape[0]
        
        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1).to(DEVICE)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Flatten
        gradients = gradients.view(batch_size, -1)
        
        # Compute penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def train_discriminator(
        self,
        real_data: torch.Tensor
    ) -> dict:
        """Train discriminator for one step"""
        batch_size = real_data.shape[0]
        
        # Generate fake data
        z = torch.randn(batch_size, GAN['latent_dim']).to(DEVICE)
        fake_data = self.generator(z).detach()
        
        # Convert to indices for discriminator
        fake_indices = torch.argmax(fake_data, dim=-1)
        
        # Discriminator outputs
        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_indices)
        
        # Wasserstein loss
        d_loss = fake_score.mean() - real_score.mean()
        
        # Gradient penalty
        gp = self.gradient_penalty(real_data.float(), fake_indices.float())
        
        # Total loss
        total_loss = d_loss + self.gp_lambda * gp
        
        # Update
        self.opt_d.zero_grad()
        total_loss.backward()
        self.opt_d.step()
        
        return {
            'd_loss': d_loss.item(),
            'gp': gp.item(),
            'real_score': real_score.mean().item(),
            'fake_score': fake_score.mean().item()
        }
    
    def train_generator(self) -> dict:
        """Train generator for one step"""
        batch_size = 64
        
        # Generate fake data
        z = torch.randn(batch_size, GAN['latent_dim']).to(DEVICE)
        fake_data = self.generator(z)
        fake_indices = torch.argmax(fake_data, dim=-1)
        
        # Discriminator score
        fake_score = self.discriminator(fake_indices)
        
        # Generator wants discriminator to think it's real
        g_loss = -fake_score.mean()
        
        # Update
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()
        
        return {
            'g_loss': g_loss.item()
        }


def create_strategy_gan():
    """Factory function to create GAN"""
    generator = BuildOrderGenerator()
    discriminator = BuildOrderDiscriminator()
    
    gan = WGAN_GP(generator, discriminator)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Created Strategy GAN:")
    print(f"  Generator parameters: {g_params:,}")
    print(f"  Discriminator parameters: {d_params:,}")
    print(f"  Device: {DEVICE}")
    
    return gan


if __name__ == "__main__":
    # Test GAN
    print("Testing Strategy GAN...")
    
    gan = create_strategy_gan()
    
    # Test generator
    print("\nTesting generator...")
    z = torch.randn(4, GAN['latent_dim']).to(DEVICE)
    fake_strategies = gan.generator(z)
    print(f"Generated strategies shape: {fake_strategies.shape}")
    print(f"Expected: [4, {GAN['build_order_length']}, {GAN['action_vocab_size']}]")
    
    # Test discriminator
    print("\nTesting discriminator...")
    fake_indices = torch.argmax(fake_strategies, dim=-1)
    scores = gan.discriminator(fake_indices)
    print(f"Discriminator scores shape: {scores.shape}")
    print(f"Scores: {scores.squeeze().tolist()}")
    
    # Test training step
    print("\nTesting training step...")
    real_data = torch.randint(0, GAN['action_vocab_size'], (4, GAN['build_order_length'])).to(DEVICE)
    
    d_stats = gan.train_discriminator(real_data)
    print(f"Discriminator stats: {d_stats}")
    
    g_stats = gan.train_generator()
    print(f"Generator stats: {g_stats}")
    
    print("\n✓ GAN test passed!")

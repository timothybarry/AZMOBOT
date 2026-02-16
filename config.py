"""
Configuration file for StarCraft 2 AI Bot
Modify these settings to customize training and bot behavior
"""

import torch
from pathlib import Path

# ============================================================================
# GENERAL SETTINGS
# ============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
REPLAYS_DIR = DATA_DIR / "replays"
MODELS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, REPLAYS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# STARCRAFT 2 SETTINGS
# ============================================================================

# Game settings
RACE = "terran"  # Options: terran, protoss, zerg, random
MAP_NAME = "Simple64"  # Training map
SCREEN_SIZE = 84  # Screen resolution (84x84 is standard)
MINIMAP_SIZE = 64  # Minimap resolution

# Action settings
APM_LIMIT = 300  # Actions per minute limit (human-like)
STEP_MULTIPLIER = 8  # Game steps per agent step (lower = more actions)

# ============================================================================
# IMITATION LEARNING SETTINGS
# ============================================================================

IMITATION = {
    # Data
    "num_replays": 5000,  # Number of replays to use for training
    "replay_min_mmr": 4000,  # Minimum MMR (Diamond level)
    "replay_race": RACE,
    "sequence_length": 128,  # Length of sequences for training
    
    # Model architecture
    "embedding_dim": 256,
    "lstm_hidden": 512,
    "lstm_layers": 3,
    "attention_heads": 8,
    "dropout": 0.1,
    
    # Training
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 100,
    "warmup_steps": 1000,
    "grad_clip": 1.0,
    
    # Optimization
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "scheduler": "cosine",
}

# ============================================================================
# REINFORCEMENT LEARNING SETTINGS
# ============================================================================

RL = {
    # Algorithm (PPO - Proximal Policy Optimization)
    "algorithm": "ppo",
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE parameter
    "clip_epsilon": 0.2,  # PPO clipping
    "vf_coef": 0.5,  # Value function coefficient
    "entropy_coef": 0.01,  # Entropy bonus
    
    # Network architecture
    "policy_hidden": [512, 512, 256],
    "value_hidden": [512, 512, 256],
    "lstm_hidden": 512,
    "lstm_layers": 2,
    
    # Training
    "learning_rate": 0.0003,
    "batch_size": 256,
    "n_steps": 2048,  # Steps per update
    "n_epochs": 10,  # Optimization epochs per update
    "max_episodes": 50000,
    
    # Reward shaping
    "reward_win": 1.0,
    "reward_loss": -1.0,
    "reward_kill_unit": 0.01,
    "reward_build_expansion": 0.05,
    "reward_resource_efficiency": 0.002,
    "reward_lose_unit": -0.005,
    
    # Exploration
    "initial_epsilon": 1.0,
    "final_epsilon": 0.1,
    "epsilon_decay": 0.995,
}

# ============================================================================
# GAN SETTINGS
# ============================================================================

GAN = {
    # Architecture
    "latent_dim": 100,  # Noise vector dimension
    "generator_hidden": [256, 512, 512, 256],
    "discriminator_hidden": [256, 512, 512, 256],
    "build_order_length": 50,  # Max build order length
    "action_vocab_size": 100,  # Number of possible actions
    
    # Training
    "batch_size": 64,
    "learning_rate_g": 0.0002,  # Generator learning rate
    "learning_rate_d": 0.0002,  # Discriminator learning rate
    "beta1": 0.5,  # Adam beta1
    "beta2": 0.999,  # Adam beta2
    "n_iterations": 10000,
    "d_steps": 5,  # Discriminator steps per generator step
    "gp_lambda": 10.0,  # Gradient penalty (if using WGAN-GP)
    
    # Loss type
    "loss_type": "wgan-gp",  # Options: "vanilla", "wgan", "wgan-gp"
}

# ============================================================================
# MULTI-AGENT LEAGUE SETTINGS
# ============================================================================

LEAGUE = {
    # League composition
    "n_main_agents": 5,  # Number of main competitive agents
    "n_main_exploiters": 2,  # Agents that exploit main agents
    "n_league_exploiters": 3,  # Agents that exploit specific strategies
    
    # Training
    "games_per_agent": 100,  # Games before updating
    "save_frequency": 1000,  # Save checkpoint every N games
    "eval_frequency": 500,  # Evaluate every N games
    
    # Opponent sampling
    "prob_current_main": 0.5,  # Probability of facing current main agent
    "prob_exploiters": 0.3,  # Probability of facing exploiters
    "prob_historical": 0.2,  # Probability of facing old versions
    
    # Diversity
    "strategy_diversity_bonus": 0.1,  # Bonus for unique strategies
    "min_games_before_exploiter": 5000,  # Games before adding exploiter
}

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

# Spatial encoder (for screen/minimap)
SPATIAL_ENCODER = {
    "type": "resnet",  # Options: "cnn", "resnet"
    "channels": [32, 64, 128],
    "kernel_sizes": [5, 3, 3],
    "strides": [2, 2, 1],
    "use_batch_norm": True,
}

# Attention mechanism
ATTENTION = {
    "type": "multi-head",  # Options: "multi-head", "scaled-dot"
    "num_heads": 8,
    "key_dim": 64,
    "value_dim": 64,
}

# Action heads
ACTION_HEADS = {
    "action_type": 573,  # Number of possible action types in SC2
    "select_point": True,  # Whether to select point on screen
    "select_unit": True,  # Whether to select specific units
}

# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

CURRICULUM = {
    "enabled": True,
    "stages": [
        {
            "name": "basic_macro",
            "episodes": 1000,
            "tasks": ["build_workers", "spend_resources"],
            "difficulty": "easy",
        },
        {
            "name": "micro_battles",
            "episodes": 2000,
            "tasks": ["unit_control", "combat"],
            "difficulty": "medium",
        },
        {
            "name": "full_game",
            "episodes": 10000,
            "tasks": ["macro", "micro", "strategy"],
            "difficulty": "hard",
        },
    ]
}

# ============================================================================
# HIERARCHICAL RL (Advanced)
# ============================================================================

HIERARCHICAL = {
    "enabled": False,  # Enable hierarchical RL
    "levels": 3,  # Number of hierarchy levels
    
    # High-level (strategy)
    "high_level": {
        "actions": ["attack", "expand", "tech", "defend"],
        "decision_frequency": 120,  # Frames between decisions (5 seconds)
    },
    
    # Mid-level (tactics)
    "mid_level": {
        "decision_frequency": 24,  # 1 second
    },
    
    # Low-level (execution)
    "low_level": {
        "decision_frequency": 1,  # Every frame
    },
}

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

LOGGING = {
    "tensorboard": True,
    "log_frequency": 10,  # Log every N episodes
    "save_replays": True,
    "save_best_only": False,
    "verbose": 1,  # 0: quiet, 1: progress bar, 2: detailed
}

# Weights & Biases (optional)
WANDB = {
    "enabled": False,  # Set to True to use W&B
    "project": "sc2-ai-bot",
    "entity": None,  # Your W&B username
}

# ============================================================================
# EVALUATION
# ============================================================================

EVALUATION = {
    "n_games": 100,  # Number of games for evaluation
    "opponents": ["Easy", "Medium", "Hard", "VeryHard", "Elite"],
    "save_replays": True,
    "detailed_stats": True,
}

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

PERFORMANCE = {
    "num_workers": 4,  # Parallel environments
    "prefetch_buffer": 2,  # Data prefetching
    "mixed_precision": True,  # Use FP16 training
    "compile_model": False,  # Torch compile (experimental)
}

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

DEBUG = {
    "enabled": False,
    "save_observations": False,
    "visualize_attention": False,
    "print_actions": False,
}

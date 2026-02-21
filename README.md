# StarCraft 2 AI Bot - Complete Implementation

This is a complete, working implementation of a StarCraft 2 AI bot using multiple ML techniques including Imitation Learning, Reinforcement Learning, and GANs.

## Project Structure

```
sc2_ai_bot/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration settings
│
├── 1_imitation_learning/              # Phase 1: Learn from replays
│   ├── replay_parser.py              # Extract data from SC2 replays
│   ├── dataset.py                    # Dataset preparation
│   ├── imitation_model.py            # Neural network architecture
│   └── train_imitation.py            # Training script
│
├── 2_reinforcement_learning/          # Phase 2: Self-play RL
│   ├── environment.py                # SC2 environment wrapper
│   ├── ppo_agent.py                  # PPO algorithm implementation
│   ├── replay_buffer.py              # Experience storage
│   └── train_rl.py                   # RL training script
│
├── 3_gan_strategies/                  # Phase 3: Strategy diversity
│   ├── strategy_gan.py               # GAN for build orders
│   ├── generator.py                  # Generator network
│   ├── discriminator.py              # Discriminator network
│   └── train_gan.py                  # GAN training script
│
├── 4_multi_agent_league/              # Phase 4: League training
│   ├── league.py                     # Multi-agent league system
│   ├── exploiter_agent.py            # Exploiter agents
│   └── train_league.py               # League training script
│
├── bot/                               # Main bot implementation
│   ├── base_bot.py                   # Base bot class
│   ├── imitation_bot.py              # Bot using imitation learning
│   ├── rl_bot.py                     # Bot using RL
│   └── hybrid_bot.py                 # Combined approach
│
├── models/                            # Neural network architectures
│   ├── lstm_model.py                 # LSTM-based architecture
│   ├── attention.py                  # Attention mechanisms
│   └── hierarchical.py               # Hierarchical RL networks
│
├── utils/                             # Utility functions
│   ├── replay_utils.py               # Replay processing
│   ├── sc2_utils.py                  # SC2 helper functions
│   ├── visualization.py              # Training visualizations
│   └── metrics.py                    # Performance metrics
│
└── scripts/                           # Executable scripts
    ├── download_replays.py           # Download replay dataset
    ├── run_bot.py                    # Run the bot in SC2
    └── evaluate.py                   # Evaluation scripts
```

## Installation

### Prerequisites
- StarCraft 2 (Free Starter Edition works)
- Python 3.9+
- 8GB+ RAM
- GPU recommended but not required

### Step 1: Install StarCraft 2
Download from: https://starcraft2.com/en-us/

### Step 2: Set up Python environment
```bash
# Create virtual environment
python -m venv sc2_env
source sc2_env/bin/activate  # On Windows: sc2_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download replays
```bash
python scripts/download_replays.py --num-replays 1000
```

## Quick Start

### Run a Simple Bot (Test Installation)
```bash
python scripts/run_bot.py --bot simple
```

### Train Imitation Learning Model
```bash
python 1_imitation_learning/train_imitation.py --epochs 50
```

### Train RL Agent 
```bash
python 2_reinforcement_learning/train_rl.py --episodes 10000
```

### Train GAN for Strategy Diversity
```bash
python 3_gan_strategies/train_gan.py --iterations 5000
```

### Run League Training
```bash
python 4_multi_agent_league/train_league.py --agents 5
```

## Progressive Training Path

Follow these steps in order:

### Phase 1: Imitation Learning (Week 1-2)
Goal: Learn basic strategies from expert replays
```bash
# 1. Download replays
python scripts/download_replays.py --num-replays 5000

# 2. Parse and prepare dataset
python 1_imitation_learning/replay_parser.py

# 3. Train imitation model
python 1_imitation_learning/train_imitation.py --epochs 100

# 4. Test the bot
python scripts/run_bot.py --bot imitation
```
**Expected Performance**: Beat Easy AI, ~Bronze-Silver level

### Phase 2: Reinforcement Learning (Week 3-6)
Goal: Improve through self-play
```bash
# 1. Train RL agent (uses imitation model as starting point)
python 2_reinforcement_learning/train_rl.py --episodes 50000

# 2. Test improved bot
python scripts/run_bot.py --bot rl
```
**Expected Performance**: Beat Medium AI, ~Gold-Platinum level

### Phase 3: GAN Strategy Diversity (Week 7-8)
Goal: Generate diverse build orders
```bash
# 1. Train GAN on build orders
python 3_gan_strategies/train_gan.py --iterations 10000

# 2. Generate new strategies
python 3_gan_strategies/generate_strategies.py --num-strategies 100

# 3. Test hybrid bot
python scripts/run_bot.py --bot hybrid
```
**Expected Performance**: More unpredictable, harder to counter

### Phase 4: League Training (Week 9-12)
Goal: Robust multi-strategy agent
```bash
# 1. Start league training
python 4_multi_agent_league/train_league.py --agents 10 --days 30

# 2. Evaluate league
python scripts/evaluate.py --league-checkpoint checkpoints/league_final.pth
```
**Expected Performance**: Beat Hard AI, ~Diamond-Master level

## Configuration

Edit `config.py` to customize:
- Neural network architectures
- Training hyperparameters
- Reward functions
- League settings

## Monitoring Training

All training scripts save:
- Checkpoints: `checkpoints/`
- Logs: `logs/`
- TensorBoard: `tensorboard --logdir=runs/`

## Evaluation

```bash
# Evaluate against built-in AI
python scripts/evaluate.py --bot rl --opponent Medium --games 100

# Evaluate multiple bots
python scripts/evaluate.py --tournament --bots imitation,rl,hybrid
```

## Troubleshooting

### Common Issues

**PySC2 not finding StarCraft 2:**
```bash
export SC2PATH="/path/to/StarCraft II"
```

**Out of memory:**
- Reduce batch size in config.py
- Use smaller replay dataset
- Enable gradient accumulation

**Training is slow:**
- Use GPU (set `device='cuda'` in config.py)
- Reduce number of game steps
- Use fewer league agents

## Project Timeline

- **Week 1-2**: Imitation Learning
- **Week 3-6**: Reinforcement Learning  
- **Week 7-8**: GAN Integration
- **Week 9-12**: League Training
- **Week 13+**: Fine-tuning & Optimization

## Expected Results

| Phase | Time | Win Rate vs AI | Rank Equivalent |
|-------|------|----------------|-----------------|
| Imitation | 2 weeks | 80% vs Easy | Bronze-Silver |
| RL | 6 weeks | 70% vs Medium | Gold-Platinum |
| GAN + League | 12 weeks | 60% vs Hard | Diamond |
| Optimized | 16+ weeks | 50% vs Elite | Master+ |

## Resources

- PySC2 Documentation: https://github.com/deepmind/pysc2
- AlphaStar Paper: https://www.nature.com/articles/s41586-019-1724-z
- SC2 AI Ladder: https://sc2ai.net/

## Contributing

This is a learning project! Feel free to:
- Experiment with different architectures
- Try new RL algorithms
- Improve reward functions
- Add new features

## License

MIT - This is for educational purposes

## Next Steps

1. Run the simple bot to test installation
2. Download replays and train imitation model
3. Experiment with RL training
4. Share your results!

Good luck! 🚀

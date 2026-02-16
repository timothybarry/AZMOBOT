"""
QUICK START GUIDE
Step-by-step instructions to get your StarCraft 2 AI bot running
"""

# ============================================================================
# INSTALLATION CHECKLIST
# ============================================================================

"""
□ Step 1: Install StarCraft 2
   - Download from: https://starcraft2.com/en-us/
   - Free Starter Edition works fine
   - Install to default location

□ Step 2: Set up Python environment
   ```bash
   cd sc2_ai_bot
   python -m venv sc2_env
   source sc2_env/bin/activate  # Windows: sc2_env\Scripts\activate
   pip install -r requirements.txt
   ```

□ Step 3: Test installation
   ```bash
   python scripts/run_bot.py --bot simple --visualize
   ```
   
   This should launch SC2 and run a simple scripted bot!
"""

# ============================================================================
# PHASE 1: IMITATION LEARNING (Week 1-2)
# ============================================================================

"""
Goal: Train a bot to mimic expert players

Step 1: Download replays
-----------------------
In a production system, you'd download actual SC2 replays.
For this demo, the dataset will be simulated.

```bash
# In production, replays come from:
# - https://github.com/Blizzard/s2client-proto
# - SC2ReplayStats.com
# - Your own games

# For now, the training script creates simulated data
python 1_imitation_learning/train_imitation.py --num-replays 1000 --epochs 50
```
 
Step 2: Watch training progress
-------------------------------
```bash
tensorboard --logdir=logs/imitation
```
Open http://localhost:6006 in your browser

Step 3: Test the trained bot
----------------------------
```bash
python scripts/run_bot.py --bot imitation --model-path checkpoints/imitation_best.pth --visualize
```

Expected Results:
- Training time: 2-4 hours (CPU) or 30-60 mins (GPU)
- Final accuracy: 60-70% action prediction
- Game performance: Can beat Easy AI
- Rank equivalent: Bronze-Silver
"""

# ============================================================================
# PHASE 2: REINFORCEMENT LEARNING (Week 3-6)
# ============================================================================

"""
Goal: Improve through self-play

Step 1: Start RL training
-------------------------
```bash
python 2_reinforcement_learning/train_rl.py --episodes 10000
```

Key points:
- Loads imitation model as starting point
- Runs self-play games
- Updates policy based on wins/losses
- Takes several days to train fully

Step 2: Monitor progress
-----------------------
```bash
tensorboard --logdir=logs/rl
```

Watch for:
- Win rate increasing
- Episode length stabilizing
- Policy loss decreasing

Step 3: Test RL bot
------------------
```bash
python scripts/run_bot.py --bot rl --model-path checkpoints/rl_best.pth --opponent Medium
```

Expected Results:
- Training time: 1-3 days
- Win rate vs Medium: 60-70%
- Rank equivalent: Gold-Platinum
"""

# ============================================================================
# PHASE 3: GAN STRATEGY DIVERSITY (Week 7-8)
# ============================================================================

"""
Goal: Generate diverse build orders

Step 1: Train GAN
----------------
```bash
python 3_gan_strategies/train_gan.py --iterations 5000
```

Step 2: Generate strategies
--------------------------
```python
from strategy_gan import create_strategy_gan

gan = create_strategy_gan()
gan.generator.load_state_dict(torch.load('checkpoints/gan_generator.pth'))

# Generate 10 unique build orders
strategies = gan.generator.generate(num_samples=10)
print(strategies)
```

Step 3: Use in hybrid bot
------------------------
```bash
python scripts/run_bot.py --bot hybrid --visualize
```

Expected Results:
- More unpredictable strategies
- Harder for opponents to counter
- Similar win rate but more variety
"""

# ============================================================================
# PHASE 4: MULTI-AGENT LEAGUE (Week 9-12)
# ============================================================================

"""
Goal: Create robust, adaptable agent

Step 1: Start league training
----------------------------
```bash
python 4_multi_agent_league/train_league.py --agents 5 --days 30
```

This runs for a long time! It:
- Trains multiple agents simultaneously
- Creates exploiter agents
- Tests agents against each other
- Saves best performers

Step 2: Monitor the league
-------------------------
```bash
tensorboard --logdir=logs/league
```

Watch:
- Win rates between agents
- Strategy diversity metrics
- Elo ratings

Step 3: Deploy best agent
------------------------
```bash
python scripts/evaluate.py --league-checkpoint checkpoints/league_final.pth
```

Expected Results:
- Training time: 1-4 weeks
- Win rate vs Hard: 60%+
- Rank equivalent: Diamond-Master
- Robust to different strategies
"""

# ============================================================================
# EXAMPLE USAGE PATTERNS
# ============================================================================

"""
Example 1: Quick test (5 minutes)
---------------------------------
python scripts/run_bot.py --bot simple --opponent Easy --visualize

Example 2: Train imitation model (1 hour)
-----------------------------------------
python 1_imitation_learning/train_imitation.py --epochs 10 --num-replays 100

Example 3: Evaluate bot (30 minutes)
------------------------------------
python scripts/run_bot.py --bot imitation --opponent Medium --num-games 50

Example 4: Compare bots
-----------------------
python scripts/evaluate.py --tournament --bots simple,imitation,rl --games 20

Example 5: Generate GAN strategies
----------------------------------
python 3_gan_strategies/generate_strategies.py --num-strategies 100 --save-to strategies.json
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: "SC2 not found"
Solution: Set environment variable
```bash
export SC2PATH="/path/to/StarCraft II"
```

Problem: Out of memory during training
Solution: Reduce batch size in config.py
```python
IMITATION['batch_size'] = 16  # Instead of 32
RL['batch_size'] = 128  # Instead of 256
```

Problem: Training is too slow
Solution: 
1. Use GPU (set DEVICE='cuda' in config.py)
2. Reduce number of replays
3. Use smaller model (reduce hidden dimensions)

Problem: Bot doesn't do anything
Solution: Check that model is loaded correctly
```python
# In your bot code
print(f"Model loaded from: {model_path}")
print(f"Model device: {next(model.parameters()).device}")
```

Problem: PySC2 errors
Solution: Make sure SC2 is running and accessible
```bash
python -m pysc2.bin.agent --map Simple64
```
"""

# ============================================================================
# CUSTOMIZATION TIPS
# ============================================================================

"""
Want to modify the bot? Here are key files:

1. Change network architecture:
   - models/lstm_model.py
   - 1_imitation_learning/imitation_model.py

2. Change reward function:
   - config.py (RL section)
   - 2_reinforcement_learning/ppo_agent.py

3. Add new strategies:
   - 3_gan_strategies/strategy_gan.py
   - Create custom generator/discriminator

4. Modify training:
   - config.py (all hyperparameters)
   - Individual train_*.py scripts

5. Change bot behavior:
   - bot/ directory
   - scripts/run_bot.py
"""

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

"""
Expected performance on consumer hardware:

GPU (RTX 3080):
- Imitation training: ~1 hour for 50 epochs
- RL training: ~12 hours for 10k episodes
- GAN training: ~2 hours for 5k iterations
- Games per second: ~100 (headless)

CPU (8-core):
- Imitation training: ~4 hours for 50 epochs
- RL training: ~2 days for 10k episodes
- GAN training: ~6 hours for 5k iterations
- Games per second: ~10 (headless)

Memory requirements:
- Imitation: 4-8GB RAM
- RL: 8-16GB RAM
- League: 16-32GB RAM
- VRAM: 6-8GB recommended
"""

# ============================================================================
# NEXT STEPS AFTER BASIC TRAINING
# ============================================================================

"""
Once you have a working bot, try:

1. Curriculum Learning
   - Train on progressively harder opponents
   - Start with micro tasks, then full games

2. Hierarchical RL
   - Add high-level strategy layer
   - Separate macro and micro decision making

3. Opponent Modeling
   - Predict opponent's strategy
   - Adapt your strategy accordingly

4. Meta-Learning
   - Quick adaptation to new opponents
   - Few-shot learning

5. Multi-Race Support
   - Train separate bots for each race
   - Or one universal bot

6. Human Imitation
   - Fine-tune on your own games
   - Copy your personal play style

7. Competition
   - Enter AI ladder: https://sc2ai.net/
   - Compete against other bots
"""

# ============================================================================
# RESOURCES AND LINKS
# ============================================================================

"""
Official Resources:
- PySC2: https://github.com/deepmind/pysc2
- SC2 API: https://github.com/Blizzard/s2client-proto
- AlphaStar: https://deepmind.com/blog/article/alphastar

Learning Resources:
- OpenAI Spinning Up: https://spinningup.openai.com/
- Lil'Log RL: https://lilianweng.github.io/posts/2018-02-19-rl-overview/
- GANs Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Communities:
- r/starcraft_ai
- SC2AI Discord
- SC2 AI Arena

Papers to Read:
1. AlphaStar (Nature 2019) - The definitive SC2 AI paper
2. PPO (Schulman 2017) - The RL algorithm we use
3. WGAN-GP (Gulrajani 2017) - Stable GAN training

Code Repositories:
- python-sc2: https://github.com/BurnySc2/python-sc2
- SC2 AI examples: https://github.com/deepmind/pysc2/tree/master/pysc2/agents
"""

# ============================================================================
# CONTACT AND SUPPORT
# ============================================================================

"""
Having issues? Here's how to get help:

1. Check the README.md first
2. Look at error messages carefully
3. Check TensorBoard logs
4. Try with simpler settings
5. Search GitHub issues
6. Ask in community Discord/Reddit

When reporting issues, include:
- Full error message
- Your config.py settings
- Hardware specs (CPU/GPU/RAM)
- Python and PyTorch versions
- What you were trying to do
"""

print(__doc__)

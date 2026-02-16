"""
Main Bot Runner
Run your trained bot in StarCraft 2
"""

import sys
import argparse
import numpy as np
import torch

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

sys.path.append('..')
from config import *


class SC2Bot(base_agent.BaseAgent):
    """
    StarCraft 2 Bot that uses trained models
    """
    
    def __init__(self, model_type='imitation', model_path=None):
        super().__init__()
        
        self.model_type = model_type
        self.device = DEVICE
        
        # Load model based on type
        if model_type == 'imitation':
            from bot.imitation_bot import ImitationBot
            self.bot = ImitationBot(model_path)
        elif model_type == 'rl':
            from bot.rl_bot import RLBot
            self.bot = RLBot(model_path)
        elif model_type == 'hybrid':
            from bot.hybrid_bot import HybridBot
            self.bot = HybridBot(model_path)
        else:  # simple scripted bot
            self.bot = SimpleScriptedBot()
        
        self.step_count = 0
        
    def step(self, obs):
        """Called every game step"""
        super().step(obs)
        self.step_count += 1
        
        # Get action from bot
        action = self.bot.step(obs)
        
        return action


class SimpleScriptedBot:
    """
    Simple scripted bot for testing
    Basic macro: builds workers, expands, makes army
    """
    
    def __init__(self):
        self.build_order = [
            "build_supply",
            "build_worker",
            "build_worker",
            "build_barracks",
            "build_worker",
            "train_marine",
            "train_marine",
            "expand",
        ]
        self.step_count = 0
        
    def step(self, obs):
        """Simple build order execution"""
        self.step_count += 1
        
        # Only act every 8 steps (reduce APM)
        if self.step_count % 8 != 0:
            return actions.FUNCTIONS.no_op()
        
        # Get game state
        player = obs.observation.player
        
        # Build supply if needed
        if player.food_used >= player.food_cap - 2:
            if self._can_build_supply(obs):
                return self._build_supply(obs)
        
        # Build workers
        if player.food_workers < 50:
            if self._can_build_worker(obs):
                return self._build_worker(obs)
        
        # Build barracks
        if len(self._get_units_by_type(obs, units.Terran.Barracks)) < 3:
            if self._can_build_barracks(obs):
                return self._build_barracks(obs)
        
        # Train marines
        if self._can_train_marine(obs):
            return self._train_marine(obs)
        
        # Attack if we have army
        marines = self._get_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 20:
            return self._attack(obs)
        
        return actions.FUNCTIONS.no_op()
    
    def _get_units_by_type(self, obs, unit_type):
        """Get units of specific type"""
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.SELF]
    
    def _can_build_supply(self, obs):
        """Check if can build supply depot"""
        return (obs.observation.player.minerals >= 100 and
                actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions)
    
    def _build_supply(self, obs):
        """Build supply depot"""
        # Find a random location near base
        x = np.random.randint(20, 40)
        y = np.random.randint(20, 40)
        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", [x, y])
    
    def _can_build_worker(self, obs):
        """Check if can build worker"""
        ccs = self._get_units_by_type(obs, units.Terran.CommandCenter)
        return (len(ccs) > 0 and
                obs.observation.player.minerals >= 50 and
                actions.FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions)
    
    def _build_worker(self, obs):
        """Build SCV"""
        return actions.FUNCTIONS.Train_SCV_quick("now")
    
    def _can_build_barracks(self, obs):
        """Check if can build barracks"""
        return (obs.observation.player.minerals >= 150 and
                actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions)
    
    def _build_barracks(self, obs):
        """Build barracks"""
        x = np.random.randint(25, 45)
        y = np.random.randint(25, 45)
        return actions.FUNCTIONS.Build_Barracks_screen("now", [x, y])
    
    def _can_train_marine(self, obs):
        """Check if can train marine"""
        return (obs.observation.player.minerals >= 50 and
                actions.FUNCTIONS.Train_Marine_quick.id in obs.observation.available_actions)
    
    def _train_marine(self, obs):
        """Train marine"""
        return actions.FUNCTIONS.Train_Marine_quick("now")
    
    def _attack(self, obs):
        """Send army to attack"""
        if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
            # Attack enemy base (assume bottom-right)
            return actions.FUNCTIONS.Attack_minimap("now", [48, 48])
        return actions.FUNCTIONS.no_op()


def run_game(args):
    """Run a StarCraft 2 game with the bot"""
    
    print(f"Starting StarCraft 2 with {args.bot} bot...")
    print(f"Map: {args.map}")
    print(f"Opponent: {args.opponent}")
    
    # Create bot
    bot = SC2Bot(model_type=args.bot, model_path=args.model_path)
    
    # Configure environment
    with sc2_env.SC2Env(
        map_name=args.map,
        players=[
            sc2_env.Agent(sc2_env.Race[RACE]),
            sc2_env.Bot(
                sc2_env.Race[args.opponent_race],
                sc2_env.Difficulty[args.opponent]
            )
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(
                screen=SCREEN_SIZE,
                minimap=MINIMAP_SIZE
            ),
            use_feature_units=True,
            use_raw_units=True,
        ),
        step_mul=STEP_MULTIPLIER,
        game_steps_per_episode=0,  # No limit
        visualize=args.visualize,
    ) as env:
        
        # Reset environment
        timesteps = env.reset()
        bot.reset()
        
        # Game loop
        game_over = False
        episode_reward = 0
        
        while not game_over:
            # Get current observation
            obs = timesteps[0]
            
            # Bot decides action
            action = bot.step(obs)
            
            # Execute action
            timesteps = env.step([action])
            
            # Check if game is over
            game_over = timesteps[0].last()
            
            # Accumulate reward
            episode_reward += timesteps[0].reward
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Game Over!")
        print(f"Result: {'VICTORY' if episode_reward > 0 else 'DEFEAT'}")
        print(f"Reward: {episode_reward}")
        print(f"Steps: {bot.step_count}")
        print(f"{'='*50}\n")
        
        return episode_reward > 0


def main(argv):
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run StarCraft 2 Bot')
    
    parser.add_argument('--bot', type=str, default='simple',
                       choices=['simple', 'imitation', 'rl', 'hybrid'],
                       help='Bot type to run')
    
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    
    parser.add_argument('--map', type=str, default=MAP_NAME,
                       help='Map to play on')
    
    parser.add_argument('--opponent', type=str, default='Medium',
                       choices=['VeryEasy', 'Easy', 'Medium', 'MediumHard', 
                               'Hard', 'Harder', 'VeryHard', 'CheatVision',
                               'CheatMoney', 'CheatInsane'],
                       help='Opponent difficulty')
    
    parser.add_argument('--opponent-race', type=str, default='random',
                       choices=['terran', 'protoss', 'zerg', 'random'],
                       help='Opponent race')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Show game window')
    
    parser.add_argument('--num-games', type=int, default=1,
                       help='Number of games to play')
    
    args = parser.parse_args(argv[1:])
    
    # Run games
    wins = 0
    for i in range(args.num_games):
        print(f"\nGame {i+1}/{args.num_games}")
        won = run_game(args)
        if won:
            wins += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Results: {wins}/{args.num_games} wins ({wins/args.num_games*100:.1f}%)")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    app.run(main)

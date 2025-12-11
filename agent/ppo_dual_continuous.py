"""
Dual-Control Continuous PPO Agent for VANET - BeaconHz + TxPower
===================================================================

This agent uses Proximal Policy Optimization (PPO) to control BOTH:
1. BeaconHz (beacon frequency): 4-12 Hz (constrained to optimal range)
2. TxPower (transmission power): 15-30 dBm

ACTION SPACE:
- BeaconHz options: [4, 6, 8, 10, 12] Hz (5 options)
- TxPower options: [15, 18, 21, 23, 26, 30] dBm (6 options)
- Total actions: 5 × 6 = 30 discrete actions

LEARNING APPROACH:
- Continuous learning (no episodes)
- PPO with Actor-Critic architecture
- Experience collection and batch updates
- On-policy learning with advantage estimation
- Dual target: optimize both communication frequency and transmission power
"""

import zmq
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import os
from datetime import datetime
from collections import deque

# Optional WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] WandB not installed. Run 'pip install wandb' for experiment tracking.")


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO
    - Actor: Outputs action probability distribution (policy)
    - Critic: Outputs state value estimate
    """
    
    def __init__(self, state_dim=10, action_dim=30, hidden_size=512):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Actor head (policy) - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )
        
        # Critic head (value function) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass through network"""
        shared_features = self.shared(x)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value
    
    def act(self, state):
        """
        Select action based on current policy
        Returns: action, log_prob, state_value
        """
        action_logits, state_value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value
    
    def evaluate(self, states, actions):
        """
        Evaluate actions for PPO update
        Returns: log_probs, state_values, entropy
        """
        action_logits, state_values = self.forward(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


class PPOMemory:
    """
    Memory buffer for storing experiences before PPO update
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def push(self, state, action, logprob, reward, state_value, done):
        """Add experience to memory"""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(done)
    
    def clear(self):
        """Clear all stored experiences"""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
    
    def __len__(self):
        return len(self.states)


class DualControlPPOAgent:
    """
    Continuous Learning PPO Agent for VANET with Dual Control
    
    Controls both:
    - BeaconHz: Communication frequency (2-12 Hz)
    - TxPower: Transmission power (15-30 dBm)
    """
    
    def __init__(self,
                 state_dim=10,           # 10 state features (including CBR)
                 learning_rate=0.0003,   # Learning rate
                 gamma=0.90,             # Discount factor
                 eps_clip=0.2,           # PPO clip parameter
                 K_epochs=20,             # Number of PPO epochs per update
                 entropy_coef=0.03,      # Entropy coefficient for exploration
                 value_coef=0.5,         # Value loss coefficient
                 update_timesteps=512):  # Update policy every N timesteps
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_timesteps = update_timesteps
        
        # DUAL ACTION SPACE
        # BeaconHz options: Constrained to optimal range 4-12 Hz
        self.beacon_options = [4, 6, 8, 10, 12]
        # TxPower options: 6 choices
        self.txpower_options = [15, 18, 21, 23, 26, 30]
        
        # Total action space: 5 × 6 = 30 actions
        self.action_dim = len(self.beacon_options) * len(self.txpower_options)
        
        # Create action mapping (action_idx -> (beaconHz, txPower))
        self.action_map = []
        for beacon in self.beacon_options:
            for txpower in self.txpower_options:
                self.action_map.append((beacon, txpower))
        
        print(f"[PPO] Action space: {self.action_dim} actions")
        print(f"[PPO] BeaconHz options: {self.beacon_options}")
        print(f"[PPO] TxPower options: {self.txpower_options}")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPO] Using device: {self.device}")
        
        # Create policy network (current policy)
        self.policy = ActorCriticNetwork(state_dim, self.action_dim).to(self.device)
        
        # Create old policy network (for PPO ratio calculation)
        self.policy_old = ActorCriticNetwork(state_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Loss function for critic
        self.MseLoss = nn.MSELoss()
        
        # Memory for collecting experiences
        self.memory = PPOMemory()
        
        # Statistics tracking
        self.total_reward = 0
        self.recent_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)
        self.step_counter = 0
        self.update_counter = 0
    
    def normalize_state(self, state_dict):
        """
        Normalize state features to 0-1 range
        
        State features (10 total):
        1. PDR (Packet Delivery Ratio) - already 0-1
        2. avgNeighbors - normalized by assumed max of 20
        3. beaconHz - normalized by max 20 Hz
        4. txPower - normalized by range 10-30 dBm
        5. numVehicles - normalized by assumed max of 100
        6. packetsReceived - normalized by typical max
        7. packetsSent - normalized by typical max
        8. throughput - normalized by typical max
        9. time - normalized by simulation time
        10. CBR (Channel Busy Ratio) - already 0-1
        """
        normalized = np.array([
            state_dict.get('PDR', 0.0),                          # 0-1 range
            state_dict.get('avgNeighbors', 0.0) / 20.0,         # 0-1 range
            state_dict.get('beaconHz', 8.0) / 20.0,             # 0-1 range
            (state_dict.get('txPower', 23.0) - 10.0) / 20.0,    # 0-1 range (10-30 dBm)
            state_dict.get('numVehicles', 10.0) / 100.0,        # 0-1 range
            state_dict.get('packetsReceived', 0.0) / 10000.0,   # 0-1 range
            state_dict.get('packetsSent', 0.0) / 1000.0,        # 0-1 range
            state_dict.get('throughput', 0.0) / 100000.0,       # 0-1 range
            state_dict.get('time', 0.0) / 200.0,                # 0-1 range
            state_dict.get('CBR', 0.0),                         # 0-1 range (already normalized)
        ], dtype=np.float32)
        
        return normalized
    
    def select_action(self, state, training=True):
        """
        Select action using policy network
        
        Returns: action_idx, log_prob, state_value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                # Sample action from policy (stochastic)
                action_idx, action_logprob, state_value = self.policy_old.act(state_tensor)
                return action_idx, action_logprob, state_value
            else:
                # Greedy action selection (deterministic)
                action_logits, _ = self.policy(state_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_idx = action_probs.argmax().item()
                return action_idx, None, None
    
    def get_action_params(self, action_idx):
        """
        Convert action index to actual parameters
        Returns: dict with beaconHz and txPower
        """
        beacon, txpower = self.action_map[action_idx]
        return {
            'beaconHz': beacon,
            'txPower': txpower
        }
    
    def store_experience(self, state, action_idx, logprob, reward, state_value, done):
        """Store experience in memory"""
        self.memory.push(state, action_idx, logprob, reward, state_value, done)
    
    def update(self):
        """
        PPO update using collected experiences
        
        Returns: average loss or None if not enough experiences
        """
        if len(self.memory) == 0:
            return None
        
        # Monte Carlo estimate of returns (discounted rewards)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), 
                                       reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Convert rewards to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Convert experiences to tensors
        old_states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        old_actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.stack(self.memory.logprobs).detach().to(self.device)
        old_state_values = torch.stack(self.memory.state_values).squeeze().detach().to(self.device)
        
        # Calculate advantages: A(s,a) = Q(s,a) - V(s) = R - V(s)
        advantages = rewards - old_state_values
        
        # Optimize policy for K epochs
        total_loss = 0
        for epoch in range(self.K_epochs):
            # Evaluate old actions with current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Calculate importance sampling ratio: π_θ(a|s) / π_θ_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Clipped surrogate loss (PPO's key innovation)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss components
            actor_loss = -torch.min(surr1, surr2).mean()              # Policy loss (clipped)
            critic_loss = self.MseLoss(state_values, rewards)         # Value function loss
            entropy_loss = -dist_entropy.mean()                       # Entropy bonus (exploration)
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / self.K_epochs
        
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory.clear()
        
        self.update_counter += 1
        
        return avg_loss
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'update_counter': self.update_counter,
            'total_reward': self.total_reward,
            'action_map': self.action_map,
        }, filepath)
        print(f"[PPO] Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_counter = checkpoint.get('step_counter', 0)
        self.update_counter = checkpoint.get('update_counter', 0)
        self.total_reward = checkpoint.get('total_reward', 0)
        print(f"[PPO] Model loaded from {filepath}")


def run_dual_control_ppo(port=5555, train=True, max_steps=1000,
                         save_interval=100, model_path=None, use_wandb=False):
    """
    Run Dual-Control PPO agent in continuous learning mode
    
    Controls both BeaconHz and TxPower simultaneously
    
    Args:
        port: ZMQ port for communication with NS3
        train: If True, agent learns. If False, agent only exploits
        max_steps: Maximum number of steps (0 = unlimited)
        save_interval: Save model every N steps
        model_path: Path to existing model to load
        use_wandb: If True, log to Weights & Biases
    """
    
    # Create agent
    agent = DualControlPPOAgent()
    
    # Load existing model if specified
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
    
    # Setup ZMQ communication with NS3
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    # Initialize WandB if requested and available
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE and train:
        try:
            wandb_run = wandb.init(
                project="exhibition",
                name=f"ppo_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "learning_rate": agent.learning_rate,
                    "gamma": agent.gamma,
                    "eps_clip": agent.eps_clip,
                    "K_epochs": agent.K_epochs,
                    "entropy_coef": agent.entropy_coef,
                    "value_coef": agent.value_coef,
                    "update_timesteps": agent.update_timesteps,
                    "state_dim": agent.state_dim,
                    "action_dim": agent.action_dim,
                    "beacon_options": agent.beacon_options,
                    "txpower_options": agent.txpower_options,
                    "max_steps": max_steps,
                    "save_interval": save_interval,
                },
                tags=["ppo", "vanet", "dual-control", "continuous-learning"]
            )
            print("[INFO] WandB logging enabled")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")
            use_wandb = False
    elif use_wandb and not WANDB_AVAILABLE:
        print("[WARNING] WandB requested but not installed. Run: pip install wandb")
        use_wandb = False
    
    print("\n" + "="*70)
    print("DUAL-CONTROL CONTINUOUS PPO AGENT FOR VANET")
    print("Controls: BeaconHz + TxPower")
    print("="*70)
    print(f"Mode: {'TRAINING (Learning continuously)' if train else 'EVALUATION (No learning)'}")
    print(f"Port: {port}")
    print(f"Action Space: {agent.action_dim} actions (6 BeaconHz × 6 TxPower)")
    print(f"BeaconHz Options: {agent.beacon_options} Hz")
    print(f"TxPower Options: {agent.txpower_options} dBm")
    print(f"Max Steps: {'Unlimited' if max_steps == 0 else max_steps}")
    print(f"Update Interval: {agent.update_timesteps} timesteps")
    print(f"Device: {agent.device}")
    print("="*70)
    print("\nWaiting for NS3 simulation to connect...\n")
    
    step = 0
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        while max_steps == 0 or step < max_steps:
            # Receive state from NS3
            msg = socket.recv()
            data = json.loads(msg.decode())
            
            # Handle non-state messages
            if data.get('type') != 'state':
                socket.send_string(json.dumps({
                    "action": {
                        "beaconHz": 8,
                        "txPower": 23
                    }
                }))
                continue
            
            # Extract and normalize state
            state_dict = data.get('data', {})
            current_state = agent.normalize_state(state_dict)
            
            # Display current status
            if step % 10 == 0 or step < 5:
                print(f"\n{'='*70}")
                print(f"Step {step + 1}")
                print(f"{'='*70}")
            
            print(f"[State] Time: {state_dict.get('time', 0):.1f}s | "
                  f"PDR: {state_dict.get('PDR', 0):.3f} | "
                  f"Neighbors: {state_dict.get('avgNeighbors', 0):.1f} | "
                  f"Throughput: {state_dict.get('throughput', 0):.0f} bps | "
                  f"CBR: {state_dict.get('CBR', 0):.3f}")
            print(f"[Current] BeaconHz: {state_dict.get('beaconHz', 0):.1f} Hz | "
                  f"TxPower: {state_dict.get('txPower', 0):.1f} dBm")
            
            # Select action
            action_idx, action_logprob, state_value = agent.select_action(current_state, training=train)
            action = agent.get_action_params(action_idx)
            
            print(f"[Action] BeaconHz: {action['beaconHz']} Hz | "
                  f"TxPower: {action['txPower']} dBm | "
                  f"(Action #{action_idx})")
            if train:
                print(f"[Learning] Memory: {len(agent.memory)}/{agent.update_timesteps} | "
                      f"Updates: {agent.update_counter}")
            
            # Send action to NS3
            socket.send_string(json.dumps({"action": action}))
            
            # Receive reward from NS3
            msg = socket.recv()
            reward_msg = json.loads(msg.decode())
            reward = reward_msg.get('reward', 0.0)
            done = reward_msg.get('done', False)
            
            print(f"[Feedback] Reward: {reward:.3f} | Done: {done}")
            
            # Update statistics
            agent.total_reward += reward
            agent.recent_rewards.append(reward)
            
            # Store experience in memory
            if train:
                agent.store_experience(current_state, action_idx, action_logprob, 
                                      reward, state_value, done)
            
            # Perform PPO update when enough experiences collected
            loss = None
            if train and len(agent.memory) >= agent.update_timesteps:
                loss = agent.update()
                if loss is not None:
                    agent.recent_losses.append(loss)
                    print(f"[Training] PPO Update #{agent.update_counter} | Loss: {loss:.4f}")
            
            # Log to WandB if enabled
            if use_wandb and wandb_run:
                log_data = {
                    "step": step,
                    "reward": reward,
                    "total_reward": agent.total_reward,
                    "memory_size": len(agent.memory),
                    "updates": agent.update_counter,
                    "chosen_beaconHz": action['beaconHz'],
                    "chosen_txPower": action['txPower'],
                    "chosen_action_idx": action_idx,
                    "state_pdr": state_dict.get('PDR', 0),
                    "state_avg_neighbors": state_dict.get('avgNeighbors', 0),
                    "state_throughput": state_dict.get('throughput', 0),
                    "state_beaconHz": state_dict.get('beaconHz', 0),
                    "state_txPower": state_dict.get('txPower', 0),
                    "state_cbr": state_dict.get('CBR', 0),
                }
                if loss is not None:
                    log_data["loss"] = loss
                
                wandb.log(log_data)
            
            # Print summary statistics periodically
            if step > 0 and step % 50 == 0:
                avg_reward = np.mean(agent.recent_rewards) if agent.recent_rewards else 0
                avg_loss = np.mean(agent.recent_losses) if agent.recent_losses else 0
                print(f"\n{'*'*70}")
                print(f"PROGRESS SUMMARY (Last 50 steps)")
                print(f"{'*'*70}")
                print(f"Total Steps: {step}")
                print(f"Total Reward: {agent.total_reward:.2f}")
                print(f"Avg Recent Reward: {avg_reward:.3f}")
                if train and agent.recent_losses:
                    print(f"Avg Recent Loss: {avg_loss:.4f}")
                print(f"PPO Updates: {agent.update_counter}")
                print(f"{'*'*70}\n")
            
            # Send acknowledgment
            socket.send_string("ack")
            
            # Save model periodically
            if train and save_interval > 0 and (step + 1) % save_interval == 0:
                model_file = f"models/ppo_dual_step{step+1}_{timestamp}.pth"
                agent.save_model(model_file)
            
            step += 1
            agent.step_counter = step
            
            # Check if simulation ended
            if done:
                print("\n[INFO] Simulation ended (done=True received)")
                # Do final update if there are experiences in memory
                if train and len(agent.memory) > 0:
                    loss = agent.update()
                    if loss is not None:
                        print(f"[Training] Final PPO Update | Loss: {loss:.4f}")
                break
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user (Ctrl+C)")
    
    finally:
        # Final save
        if train:
            final_model = f"models/ppo_dual_final_{timestamp}.pth"
            agent.save_model(final_model)
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total Steps: {step}")
        print(f"Total Reward: {agent.total_reward:.2f}")
        if agent.recent_rewards:
            print(f"Average Recent Reward (last 100): {np.mean(agent.recent_rewards):.3f}")
            print(f"Max Recent Reward: {np.max(agent.recent_rewards):.3f}")
            print(f"Min Recent Reward: {np.min(agent.recent_rewards):.3f}")
        if train and agent.recent_losses:
            print(f"Average Recent Loss (last 100): {np.mean(agent.recent_losses):.4f}")
        print(f"Total PPO Updates: {agent.update_counter}")
        print(f"Final Memory Size: {len(agent.memory)}")
        print("="*70)
        
        # Close WandB if it was initialized
        if use_wandb and wandb_run:
            wandb.finish()
            print("[INFO] WandB run finished")
        
        socket.close()
        ctx.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dual-Control Continuous PPO Agent for VANET (BeaconHz + TxPower)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training mode (continuous learning with dual control)
  python ppo_dual_continuous.py --train --max_steps 500
  
  # Evaluation mode (no learning, uses learned policy)
  python ppo_dual_continuous.py --eval --model models/ppo_dual_final_XXX.pth
  
  # Unlimited continuous learning (until Ctrl+C)
  python ppo_dual_continuous.py --train --max_steps 0
  
  # Training with WandB logging
  python ppo_dual_continuous.py --train --max_steps 1000 --wandb
  
  # Custom update interval (collect more experiences before update)
  python ppo_dual_continuous.py --train --max_steps 1000 --update_interval 256
        """
    )
    
    parser.add_argument('--port', type=int, default=5555,
                       help='ZMQ port for NS3 communication (default: 5555)')
    parser.add_argument('--train', action='store_true',
                       help='Enable training mode (agent learns continuously)')
    parser.add_argument('--eval', action='store_true',
                       help='Enable evaluation mode (no learning, exploitation only)')
    parser.add_argument('--max_steps', type=int, default=0,
                       help='Maximum steps (0=unlimited, default: 0)')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save model every N steps (default: 100)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file to load')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging (requires: pip install wandb)')
    parser.add_argument('--update_interval', type=int, default=128,
                       help='PPO update interval in timesteps (default: 128)')
    
    args = parser.parse_args()
    
    # Default to training mode if neither specified
    train_mode = args.train or not args.eval
    
    # Create agent with custom update interval if specified
    run_dual_control_ppo(
        port=args.port,
        train=train_mode,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        model_path=args.model,
        use_wandb=args.wandb
    )

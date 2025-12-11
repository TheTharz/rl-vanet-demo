"""
Model Testing Script with WandB Integration
============================================

Test any trained RL model (PPO or DQN) with:
- WandB logging support
- Detailed performance metrics
- Summary statistics
- CSV export
- Visualization plots
"""

import argparse
import os
import sys
import json
import zmq
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# Optional WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] WandB not installed. Run 'pip install wandb' for experiment tracking.")


class ModelTester:
    """Universal tester for PPO and DQN models"""
    
    def __init__(self, model_path, model_type='auto', port=5555, use_wandb=False, 
                 wandb_project="vanet-model-testing", wandb_tags=None):
        """
        Initialize model tester
        
        Args:
            model_path: Path to trained model file (.pth)
            model_type: 'ppo_dual', 'dqn_dual', 'dqn_simple', or 'auto' (auto-detect)
            port: ZMQ port for NS3 communication
            use_wandb: Enable WandB logging
            wandb_project: WandB project name
            wandb_tags: Additional tags for WandB run
        """
        self.model_path = model_path
        self.port = port
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            sys.exit(1)
        
        # Auto-detect model type
        if model_type == 'auto':
            if 'ppo' in model_path.lower():
                model_type = 'ppo_dual'
            elif 'dqn_dual' in model_path.lower():
                model_type = 'dqn_dual'
            elif 'dqn' in model_path.lower():
                model_type = 'dqn_simple'
            else:
                print("[ERROR] Could not auto-detect model type. Please specify --model_type")
                sys.exit(1)
        
        self.model_type = model_type
        print(f"[INFO] Detected model type: {model_type}")
        
        # Load appropriate agent
        self.agent = self._load_agent()
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.step_data = []
        
        # Setup ZMQ
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Initialize WandB if enabled
        self.wandb_run = None
        if self.use_wandb:
            try:
                tags = ['testing', 'evaluation', model_type]
                if wandb_tags:
                    tags.extend(wandb_tags)
                
                model_name = os.path.basename(model_path).replace('.pth', '')
                
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=f"test_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model_path": model_path,
                        "model_type": model_type,
                        "mode": "evaluation",
                        "port": port,
                    },
                    tags=tags
                )
                print("[INFO] WandB logging enabled")
            except Exception as e:
                print(f"[WARNING] Failed to initialize WandB: {e}")
                self.use_wandb = False
        
        print(f"[INFO] Agent loaded and ready on port {port}")
    
    def _load_agent(self):
        """Load the appropriate agent based on model type"""
        if self.model_type == 'ppo_dual':
            from ppo_dual_continuous import DualControlPPOAgent
            agent = DualControlPPOAgent()
            agent.load_model(self.model_path)
            print(f"[LOADED] PPO Dual Control Agent")
            return agent
        
        elif self.model_type == 'dqn_dual':
            from dqn_dual_continuous import DualControlDQNAgent
            agent = DualControlDQNAgent()
            agent.load_model(self.model_path)
            print(f"[LOADED] DQN Dual Control Agent")
            return agent
        
        elif self.model_type == 'dqn_simple':
            from dqn_simple_continuous import ContinuousDQNAgent
            agent = ContinuousDQNAgent()
            agent.load_model(self.model_path)
            print(f"[LOADED] DQN Simple Agent (BeaconHz only)")
            return agent
        
        else:
            print(f"[ERROR] Unknown model type: {self.model_type}")
            sys.exit(1)
    
    def run_test(self, max_steps=500, verbose=True, log_interval=10):
        """
        Run evaluation test
        
        Args:
            max_steps: Maximum number of steps to run
            verbose: Print detailed output
            log_interval: Print stats every N steps
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION TEST")
        print("="*80)
        print(f"Model:      {os.path.basename(self.model_path)}")
        print(f"Type:       {self.model_type.upper()}")
        print(f"Max Steps:  {max_steps}")
        print(f"Mode:       EVALUATION (No Learning)")
        print(f"WandB:      {'Enabled' if self.use_wandb else 'Disabled'}")
        print("="*80)
        print("\nWaiting for NS3 simulation...\n")
        
        step = 0
        total_reward = 0
        
        # Action tracking
        action_counts = defaultdict(int)
        beacon_counts = defaultdict(int)
        txpower_counts = defaultdict(int)
        
        # Performance windows
        window_size = 50
        recent_rewards = []
        recent_pdrs = []
        recent_throughputs = []
        
        try:
            while step < max_steps:
                # Receive state
                msg = self.socket.recv()
                data = json.loads(msg.decode())
                
                # Handle non-state messages
                if data.get('type') != 'state':
                    default_action = {"beaconHz": 8, "txPower": 23}
                    self.socket.send_string(json.dumps({"action": default_action}))
                    continue
                
                # Extract and normalize state
                state_dict = data.get('data', {})
                current_state = self.agent.normalize_state(state_dict)
                
                # Select action (greedy - no exploration)
                if self.model_type == 'ppo_dual':
                    action_idx, _, _ = self.agent.select_action(current_state, training=False)
                else:  # DQN
                    action_idx = self.agent.select_action(current_state, training=False)
                
                action = self.agent.get_action_params(action_idx)
                
                # Track actions
                action_counts[action_idx] += 1
                beacon_counts[action.get('beaconHz', 0)] += 1
                txpower_counts[action.get('txPower', 0)] += 1
                
                # Display progress
                if verbose and (step % log_interval == 0 or step < 5):
                    print(f"\n{'='*80}")
                    print(f"Step {step + 1}/{max_steps}")
                    print(f"{'='*80}")
                    print(f"[State]")
                    print(f"  Time:       {state_dict.get('time', 0):.1f}s")
                    print(f"  PDR:        {state_dict.get('PDR', 0):.4f}")
                    print(f"  Neighbors:  {state_dict.get('avgNeighbors', 0):.2f}")
                    print(f"  Throughput: {state_dict.get('throughput', 0):.0f} bps")
                    print(f"  CBR:        {state_dict.get('CBR', 0):.4f}")
                    print(f"[Action]")
                    print(f"  BeaconHz:   {action['beaconHz']} Hz")
                    print(f"  TxPower:    {action['txPower']} dBm")
                
                # Send action to NS3
                self.socket.send_string(json.dumps({"action": action}))
                
                # Receive reward
                msg = self.socket.recv()
                reward_msg = json.loads(msg.decode())
                reward = reward_msg.get('reward', 0.0)
                done = reward_msg.get('done', False)
                
                total_reward += reward
                
                # Update windowed metrics
                recent_rewards.append(reward)
                recent_pdrs.append(state_dict.get('PDR', 0))
                recent_throughputs.append(state_dict.get('throughput', 0))
                if len(recent_rewards) > window_size:
                    recent_rewards.pop(0)
                    recent_pdrs.pop(0)
                    recent_throughputs.pop(0)
                
                # Store metrics
                step_metrics = {
                    'step': step,
                    'time': state_dict.get('time', 0),
                    'pdr': state_dict.get('PDR', 0),
                    'throughput': state_dict.get('throughput', 0),
                    'neighbors': state_dict.get('avgNeighbors', 0),
                    'cbr': state_dict.get('CBR', 0),
                    'packets_received': state_dict.get('packetsReceived', 0),
                    'packets_sent': state_dict.get('packetsSent', 0),
                    'num_vehicles': state_dict.get('numVehicles', 0),
                    'reward': reward,
                    'total_reward': total_reward,
                    'beaconHz': action['beaconHz'],
                    'txPower': action['txPower'],
                    'action_idx': action_idx,
                }
                
                self.step_data.append(step_metrics)
                
                # Store in metrics dict for easy access
                for key, value in step_metrics.items():
                    if key != 'step':
                        self.metrics[key].append(value)
                
                # Log to WandB
                if self.use_wandb and self.wandb_run:
                    wandb_log = step_metrics.copy()
                    wandb_log['avg_reward_window'] = np.mean(recent_rewards)
                    wandb_log['avg_pdr_window'] = np.mean(recent_pdrs)
                    wandb_log['avg_throughput_window'] = np.mean(recent_throughputs)
                    wandb.log(wandb_log)
                
                # Display reward
                if verbose and (step % log_interval == 0 or step < 5):
                    print(f"[Reward]")
                    print(f"  Current:    {reward:.4f}")
                    print(f"  Total:      {total_reward:.2f}")
                    print(f"  Avg (last {len(recent_rewards)}): {np.mean(recent_rewards):.4f}")
                
                # Send ack
                self.socket.send_string("ack")
                
                step += 1
                
                if done:
                    print("\n[INFO] Simulation completed (done=True)")
                    break
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Test interrupted by user (Ctrl+C)")
        
        finally:
            self.socket.close()
            self.ctx.term()
        
        # Calculate comprehensive statistics
        results = self._calculate_statistics(step, total_reward, action_counts, 
                                             beacon_counts, txpower_counts)
        
        return results
    
    def _calculate_statistics(self, total_steps, total_reward, action_counts, 
                             beacon_counts, txpower_counts):
        """Calculate comprehensive statistics from test run"""
        
        # Basic stats
        results = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'avg_reward': total_reward / total_steps if total_steps > 0 else 0,
        }
        
        # Calculate statistics for each metric
        for metric_name in ['pdr', 'throughput', 'neighbors', 'cbr', 'reward',
                           'packets_received', 'packets_sent', 'num_vehicles']:
            if metric_name in self.metrics and self.metrics[metric_name]:
                data = np.array(self.metrics[metric_name])
                results[f'avg_{metric_name}'] = float(np.mean(data))
                results[f'std_{metric_name}'] = float(np.std(data))
                results[f'min_{metric_name}'] = float(np.min(data))
                results[f'max_{metric_name}'] = float(np.max(data))
                results[f'median_{metric_name}'] = float(np.median(data))
        
        # Action distributions
        results['action_distribution'] = dict(action_counts)
        results['beacon_distribution'] = dict(beacon_counts)
        results['txpower_distribution'] = dict(txpower_counts)
        
        # Action diversity (entropy)
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_probs = [count / total_actions for count in action_counts.values()]
            results['action_entropy'] = float(-sum(p * np.log(p + 1e-10) for p in action_probs))
        else:
            results['action_entropy'] = 0.0
        
        # Most common actions
        if action_counts:
            most_common_action = max(action_counts.items(), key=lambda x: x[1])
            results['most_common_action_idx'] = int(most_common_action[0])
            results['most_common_action_count'] = int(most_common_action[1])
            results['most_common_action_pct'] = float(most_common_action[1] / total_steps * 100)
        
        if beacon_counts:
            most_common_beacon = max(beacon_counts.items(), key=lambda x: x[1])
            results['most_common_beaconHz'] = int(most_common_beacon[0])
            results['most_common_beacon_pct'] = float(most_common_beacon[1] / total_steps * 100)
        
        if txpower_counts:
            most_common_txpower = max(txpower_counts.items(), key=lambda x: x[1])
            results['most_common_txPower'] = int(most_common_txpower[0])
            results['most_common_txpower_pct'] = float(most_common_txpower[1] / total_steps * 100)
        
        return results
    
    def print_summary(self, results):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n{'Model Information':^80}")
        print("-" * 80)
        print(f"  Model Path:     {os.path.basename(results['model_path'])}")
        print(f"  Model Type:     {results['model_type'].upper()}")
        print(f"  Total Steps:    {results['total_steps']}")
        
        print(f"\n{'Reward Statistics':^80}")
        print("-" * 80)
        print(f"  Total Reward:   {results['total_reward']:.2f}")
        print(f"  Average:        {results['avg_reward']:.4f}")
        print(f"  Std Dev:        {results.get('std_reward', 0):.4f}")
        print(f"  Min:            {results.get('min_reward', 0):.4f}")
        print(f"  Max:            {results.get('max_reward', 0):.4f}")
        print(f"  Median:         {results.get('median_reward', 0):.4f}")
        
        print(f"\n{'Network Performance Metrics':^80}")
        print("-" * 80)
        print(f"  PDR (Packet Delivery Ratio):")
        print(f"    Average:      {results.get('avg_pdr', 0):.4f}")
        print(f"    Std Dev:      {results.get('std_pdr', 0):.4f}")
        print(f"    Range:        [{results.get('min_pdr', 0):.4f}, {results.get('max_pdr', 0):.4f}]")
        
        print(f"\n  Throughput (bps):")
        print(f"    Average:      {results.get('avg_throughput', 0):.0f}")
        print(f"    Std Dev:      {results.get('std_throughput', 0):.0f}")
        print(f"    Range:        [{results.get('min_throughput', 0):.0f}, {results.get('max_throughput', 0):.0f}]")
        
        print(f"\n  CBR (Channel Busy Ratio):")
        print(f"    Average:      {results.get('avg_cbr', 0):.4f}")
        print(f"    Std Dev:      {results.get('std_cbr', 0):.4f}")
        print(f"    Range:        [{results.get('min_cbr', 0):.4f}, {results.get('max_cbr', 0):.4f}]")
        
        print(f"\n  Neighbors:")
        print(f"    Average:      {results.get('avg_neighbors', 0):.2f}")
        print(f"    Std Dev:      {results.get('std_neighbors', 0):.2f}")
        
        print(f"\n{'Action Analysis':^80}")
        print("-" * 80)
        print(f"  Action Diversity (Entropy): {results.get('action_entropy', 0):.4f}")
        print(f"  Most Common Action:         #{results.get('most_common_action_idx', 'N/A')} "
              f"({results.get('most_common_action_pct', 0):.1f}%)")
        
        print(f"\n  BeaconHz Distribution:")
        beacon_dist = results.get('beacon_distribution', {})
        for beacon in sorted(beacon_dist.keys()):
            count = beacon_dist[beacon]
            pct = 100 * count / results['total_steps']
            bar = '█' * int(pct / 2)
            print(f"    {beacon:2.0f} Hz: {count:5d} times ({pct:5.1f}%) {bar}")
        print(f"  Most Common:  {results.get('most_common_beaconHz', 'N/A')} Hz "
              f"({results.get('most_common_beacon_pct', 0):.1f}%)")
        
        print(f"\n  TxPower Distribution:")
        txpower_dist = results.get('txpower_distribution', {})
        for txpower in sorted(txpower_dist.keys()):
            count = txpower_dist[txpower]
            pct = 100 * count / results['total_steps']
            bar = '█' * int(pct / 2)
            print(f"    {txpower:2.0f} dBm: {count:5d} times ({pct:5.1f}%) {bar}")
        print(f"  Most Common:  {results.get('most_common_txPower', 'N/A')} dBm "
              f"({results.get('most_common_txpower_pct', 0):.1f}%)")
        
        print("\n" + "="*80)
    
    def save_results(self, results, output_dir="test_results"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.model_path).replace('.pth', '')
        
        # Save JSON summary
        json_file = os.path.join(output_dir, f"summary_{model_name}_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] Summary: {json_file}")
        
        # Save detailed CSV
        csv_file = os.path.join(output_dir, f"details_{model_name}_{timestamp}.csv")
        with open(csv_file, 'w') as f:
            # Header
            if self.step_data:
                headers = list(self.step_data[0].keys())
                f.write(','.join(headers) + '\n')
                
                # Data rows
                for step_data in self.step_data:
                    values = [str(step_data[h]) for h in headers]
                    f.write(','.join(values) + '\n')
        
        print(f"[SAVED] Details: {csv_file}")
        
        return json_file, csv_file
    
    def generate_plots(self, results, output_dir="test_results", show=False):
        """Generate comprehensive visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.model_path).replace('.pth', '')
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Model Test Results: {model_name}', fontsize=16, fontweight='bold')
        
        # Row 1: PDR, Throughput, CBR
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.metrics['pdr'], 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=results['avg_pdr'], color='r', linestyle='--', 
                   label=f"Avg: {results['avg_pdr']:.4f}")
        ax1.set_title('Packet Delivery Ratio (PDR)', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('PDR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.metrics['throughput'], 'g-', alpha=0.7, linewidth=1)
        ax2.axhline(y=results['avg_throughput'], color='r', linestyle='--',
                   label=f"Avg: {results['avg_throughput']:.0f}")
        ax2.set_title('Throughput', fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Throughput (bps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.metrics['cbr'], 'c-', alpha=0.7, linewidth=1)
        ax3.axhline(y=results['avg_cbr'], color='r', linestyle='--',
                   label=f"Avg: {results['avg_cbr']:.4f}")
        ax3.set_title('Channel Busy Ratio (CBR)', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('CBR')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Row 2: Reward, Neighbors, Packets
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.metrics['reward'], 'm-', alpha=0.7, linewidth=1)
        ax4.axhline(y=results['avg_reward'], color='r', linestyle='--',
                   label=f"Avg: {results['avg_reward']:.4f}")
        ax4.set_title('Reward per Step', fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.metrics['neighbors'], 'orange', alpha=0.7, linewidth=1)
        ax5.axhline(y=results['avg_neighbors'], color='r', linestyle='--',
                   label=f"Avg: {results['avg_neighbors']:.2f}")
        ax5.set_title('Average Neighbors', fontweight='bold')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Neighbors')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.metrics['total_reward'], 'purple', alpha=0.7, linewidth=1.5)
        ax6.set_title('Cumulative Reward', fontweight='bold')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Total Reward')
        ax6.grid(True, alpha=0.3)
        
        # Row 3: Actions
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(self.metrics['beaconHz'], 'brown', alpha=0.7, marker='.', markersize=3, linestyle='-')
        ax7.set_title('BeaconHz Actions', fontweight='bold')
        ax7.set_xlabel('Step')
        ax7.set_ylabel('BeaconHz (Hz)')
        ax7.grid(True, alpha=0.3)
        if hasattr(self.agent, 'beacon_options'):
            ax7.set_yticks(self.agent.beacon_options)
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(self.metrics['txPower'], 'olive', alpha=0.7, marker='.', markersize=3, linestyle='-')
        ax8.set_title('TxPower Actions', fontweight='bold')
        ax8.set_xlabel('Step')
        ax8.set_ylabel('TxPower (dBm)')
        ax8.grid(True, alpha=0.3)
        if hasattr(self.agent, 'txpower_options'):
            ax8.set_yticks(self.agent.txpower_options)
        
        ax9 = fig.add_subplot(gs[2, 2])
        action_dist = results['action_distribution']
        actions = sorted(action_dist.keys())
        counts = [action_dist[a] for a in actions]
        ax9.bar(actions, counts, color='teal', alpha=0.7)
        ax9.set_title('Action Distribution', fontweight='bold')
        ax9.set_xlabel('Action Index')
        ax9.set_ylabel('Count')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Row 4: Distribution histograms
        ax10 = fig.add_subplot(gs[3, 0])
        beacon_dist = results['beacon_distribution']
        beacons = sorted(beacon_dist.keys())
        beacon_counts = [beacon_dist[b] for b in beacons]
        ax10.bar(beacons, beacon_counts, color='coral', alpha=0.7, width=0.8)
        ax10.set_title('BeaconHz Distribution', fontweight='bold')
        ax10.set_xlabel('BeaconHz (Hz)')
        ax10.set_ylabel('Count')
        ax10.grid(True, alpha=0.3, axis='y')
        
        ax11 = fig.add_subplot(gs[3, 1])
        txpower_dist = results['txpower_distribution']
        txpowers = sorted(txpower_dist.keys())
        txpower_counts = [txpower_dist[t] for t in txpowers]
        ax11.bar(txpowers, txpower_counts, color='steelblue', alpha=0.7, width=1.0)
        ax11.set_title('TxPower Distribution', fontweight='bold')
        ax11.set_xlabel('TxPower (dBm)')
        ax11.set_ylabel('Count')
        ax11.grid(True, alpha=0.3, axis='y')
        
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.hist(self.metrics['pdr'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax12.axvline(x=results['avg_pdr'], color='r', linestyle='--', linewidth=2,
                    label=f"Mean: {results['avg_pdr']:.4f}")
        ax12.set_title('PDR Distribution', fontweight='bold')
        ax12.set_xlabel('PDR')
        ax12.set_ylabel('Frequency')
        ax12.legend()
        ax12.grid(True, alpha=0.3, axis='y')
        
        # Save plot
        plot_file = os.path.join(output_dir, f"plots_{model_name}_{timestamp}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Plots: {plot_file}")
        
        # Log to WandB if enabled
        if self.use_wandb and self.wandb_run:
            wandb.log({"test_results_plot": wandb.Image(plot_file)})
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return plot_file
    
    def cleanup(self):
        """Cleanup WandB if needed"""
        if self.use_wandb and self.wandb_run:
            wandb.finish()
            print("[INFO] WandB run finished")


def main():
    parser = argparse.ArgumentParser(
        description='Test RL Model with WandB Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific PPO model
  python test_model.py --model models/ppo_dual_final_20251112_031440.pth
  
  # Test with WandB logging
  python test_model.py --model models/ppo_dual_final_20251112_031440.pth --wandb
  
  # Test DQN model with all features
  python test_model.py --model models/dqn_dual_final_20251108_050303.pth \\
      --wandb --save --plot --max_steps 1000
  
  # Test and display plots
  python test_model.py --model models/ppo_dual_step500_XXX.pth --plot --show
  
  # Specify model type explicitly
  python test_model.py --model models/my_model.pth --model_type ppo_dual
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pth)')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'ppo_dual', 'dqn_dual', 'dqn_simple'],
                       help='Model type (default: auto-detect)')
    parser.add_argument('--port', type=int, default=5555,
                       help='ZMQ port for NS3 communication (default: 5555)')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum test steps (default: 500)')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Print stats every N steps (default: 10)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON and CSV files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--show', action='store_true',
                       help='Display plots (use with --plot)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='vanet-model-testing',
                       help='WandB project name (default: vanet-model-testing)')
    parser.add_argument('--wandb_tags', type=str, nargs='+',
                       help='Additional WandB tags')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check if WandB is requested but not available
    if args.wandb and not WANDB_AVAILABLE:
        print("[ERROR] WandB requested but not installed. Run: pip install wandb")
        print("[INFO] Continuing without WandB...")
        args.wandb = False
    
    # Create tester
    tester = ModelTester(
        model_path=args.model,
        model_type=args.model_type,
        port=args.port,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags
    )
    
    # Run test
    results = tester.run_test(
        max_steps=args.max_steps,
        verbose=not args.quiet,
        log_interval=args.log_interval
    )
    
    # Print summary
    tester.print_summary(results)
    
    # Save results if requested
    if args.save:
        tester.save_results(results, output_dir=args.output_dir)
    
    # Generate plots if requested
    if args.plot:
        tester.generate_plots(results, output_dir=args.output_dir, show=args.show)
    
    # Log summary to WandB
    if args.wandb and tester.wandb_run:
        wandb.log({
            "test_summary/total_reward": results['total_reward'],
            "test_summary/avg_reward": results['avg_reward'],
            "test_summary/avg_pdr": results['avg_pdr'],
            "test_summary/avg_throughput": results['avg_throughput'],
            "test_summary/avg_cbr": results['avg_cbr'],
            "test_summary/action_entropy": results.get('action_entropy', 0),
        })
    
    # Cleanup
    tester.cleanup()
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Baseline Testing Script (No RL Agent)
======================================

Test NS3 VANET performance with fixed BeaconHz and TxPower.
No RL agent - just provides constant actions.

Useful for:
- Establishing baseline performance
- Comparing RL agent improvements
- Testing different fixed configurations
"""

import argparse
import os
import sys
import json
import zmq
import numpy as np
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


class BaselineTester:
    """Baseline tester with fixed BeaconHz and TxPower (no RL)"""
    
    def __init__(self, beaconHz, txPower, port=5555, use_wandb=False, 
                 wandb_project="vanet-baseline-testing", wandb_tags=None, run_name=None):
        """
        Initialize baseline tester
        
        Args:
            beaconHz: Fixed beacon frequency (Hz)
            txPower: Fixed transmission power (dBm)
            port: ZMQ port for NS3 communication
            use_wandb: Enable WandB logging
            wandb_project: WandB project name
            wandb_tags: Additional tags for WandB run
            run_name: Custom name for the run
        """
        self.beaconHz = beaconHz
        self.txPower = txPower
        self.port = port
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
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
                tags = ['baseline', 'no-rl', 'fixed-params']
                if wandb_tags:
                    tags.extend(wandb_tags)
                
                if run_name is None:
                    run_name = f"baseline_{beaconHz}Hz_{txPower}dBm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config={
                        "beaconHz": beaconHz,
                        "txPower": txPower,
                        "mode": "baseline",
                        "agent": "none",
                        "port": port,
                    },
                    tags=tags
                )
                print("[INFO] WandB logging enabled")
            except Exception as e:
                print(f"[WARNING] Failed to initialize WandB: {e}")
                self.use_wandb = False
        
        print(f"[INFO] Baseline tester ready on port {port}")
    
    def run_test(self, max_steps=500, verbose=True, log_interval=10):
        """
        Run baseline test with fixed parameters
        
        Args:
            max_steps: Maximum number of steps to run
            verbose: Print detailed output
            log_interval: Print stats every N steps
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*80)
        print("BASELINE TEST (NO RL AGENT)")
        print("="*80)
        print(f"BeaconHz:   {self.beaconHz} Hz (FIXED)")
        print(f"TxPower:    {self.txPower} dBm (FIXED)")
        print(f"Max Steps:  {max_steps}")
        print(f"Mode:       BASELINE (No Learning, No Adaptation)")
        print(f"WandB:      {'Enabled' if self.use_wandb else 'Disabled'}")
        print("="*80)
        print("\nWaiting for NS3 simulation...\n")
        
        step = 0
        total_reward = 0
        
        # Performance windows
        window_size = 50
        recent_rewards = []
        recent_pdrs = []
        recent_throughputs = []
        recent_cbrs = []
        
        try:
            while step < max_steps:
                # Receive state
                msg = self.socket.recv()
                data = json.loads(msg.decode())
                
                # Handle non-state messages
                if data.get('type') != 'state':
                    action = {"beaconHz": self.beaconHz, "txPower": self.txPower}
                    self.socket.send_string(json.dumps({"action": action}))
                    continue
                
                # Extract state
                state_dict = data.get('data', {})
                
                # Display progress
                if verbose and (step % log_interval == 0 or step < 5):
                    print(f"\n{'='*80}")
                    print(f"Step {step + 1}/{max_steps}")
                    print(f"{'='*80}")
                    print(f"[State]")
                    print(f"  Time:          {state_dict.get('time', 0):.1f}s")
                    print(f"  PDR:           {state_dict.get('PDR', 0):.4f}")
                    print(f"  Neighbors:     {state_dict.get('avgNeighbors', 0):.2f}")
                    print(f"  Throughput:    {state_dict.get('throughput', 0):.0f} bps")
                    print(f"  CBR:           {state_dict.get('CBR', 0):.4f}")
                    print(f"  NumVehicles:   {state_dict.get('numVehicles', 0)}")
                    print(f"  PktReceived:   {state_dict.get('packetsReceived', 0)}")
                    print(f"  PktSent:       {state_dict.get('packetsSent', 0)}")
                
                # Always send the same fixed action
                action = {"beaconHz": self.beaconHz, "txPower": self.txPower}
                
                if verbose and (step % log_interval == 0 or step < 5):
                    print(f"[Action - FIXED]")
                    print(f"  BeaconHz:      {action['beaconHz']} Hz")
                    print(f"  TxPower:       {action['txPower']} dBm")
                
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
                recent_cbrs.append(state_dict.get('CBR', 0))
                if len(recent_rewards) > window_size:
                    recent_rewards.pop(0)
                    recent_pdrs.pop(0)
                    recent_throughputs.pop(0)
                    recent_cbrs.pop(0)
                
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
                    'beaconHz': self.beaconHz,
                    'txPower': self.txPower,
                }
                
                self.step_data.append(step_metrics)
                
                # Store in metrics dict for easy access
                for key, value in step_metrics.items():
                    if key != 'step':
                        self.metrics[key].append(value)
                
                # Log to WandB
                if self.use_wandb and self.wandb_run:
                    wandb_log = step_metrics.copy()
                    wandb_log['avg_reward_window'] = np.mean(recent_rewards) if recent_rewards else 0
                    wandb_log['avg_pdr_window'] = np.mean(recent_pdrs) if recent_pdrs else 0
                    wandb_log['avg_throughput_window'] = np.mean(recent_throughputs) if recent_throughputs else 0
                    wandb_log['avg_cbr_window'] = np.mean(recent_cbrs) if recent_cbrs else 0
                    wandb.log(wandb_log)
                
                # Display reward and window averages
                if verbose and (step % log_interval == 0 or step < 5):
                    print(f"[Reward]")
                    print(f"  Current:       {reward:.4f}")
                    print(f"  Total:         {total_reward:.2f}")
                    if recent_rewards:
                        print(f"  Avg (last {len(recent_rewards)}): {np.mean(recent_rewards):.4f}")
                    if recent_pdrs:
                        print(f"[Window Averages (last {len(recent_pdrs)} steps)]")
                        print(f"  PDR:           {np.mean(recent_pdrs):.4f}")
                        print(f"  Throughput:    {np.mean(recent_throughputs):.0f} bps")
                        print(f"  CBR:           {np.mean(recent_cbrs):.4f}")
                
                # Send ack
                self.socket.send_string("ack")
                
                step += 1
                
                if done:
                    print("\n[INFO] Simulation completed (done=True)")
                    break
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Test interrupted by user (Ctrl+C)")
        
        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.socket.close()
            self.ctx.term()
        
        # Calculate comprehensive statistics
        results = self._calculate_statistics(step, total_reward)
        
        return results
    
    def _calculate_statistics(self, total_steps, total_reward):
        """Calculate comprehensive statistics from test run"""
        
        # Basic stats
        results = {
            'beaconHz': self.beaconHz,
            'txPower': self.txPower,
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
        
        # Stability metrics (coefficient of variation)
        if 'std_pdr' in results and results['avg_pdr'] > 0:
            results['cv_pdr'] = results['std_pdr'] / results['avg_pdr']
        if 'std_throughput' in results and results['avg_throughput'] > 0:
            results['cv_throughput'] = results['std_throughput'] / results['avg_throughput']
        if 'std_cbr' in results and results['avg_cbr'] > 0:
            results['cv_cbr'] = results['std_cbr'] / results['avg_cbr']
        
        return results
    
    def print_summary(self, results):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("BASELINE TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n{'Configuration':^80}")
        print("-" * 80)
        print(f"  BeaconHz:       {results['beaconHz']} Hz (FIXED)")
        print(f"  TxPower:        {results['txPower']} dBm (FIXED)")
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
        print(f"    CV:           {results.get('cv_pdr', 0):.4f}")
        print(f"    Range:        [{results.get('min_pdr', 0):.4f}, {results.get('max_pdr', 0):.4f}]")
        
        print(f"\n  Throughput (bps):")
        print(f"    Average:      {results.get('avg_throughput', 0):.0f}")
        print(f"    Std Dev:      {results.get('std_throughput', 0):.0f}")
        print(f"    CV:           {results.get('cv_throughput', 0):.4f}")
        print(f"    Range:        [{results.get('min_throughput', 0):.0f}, {results.get('max_throughput', 0):.0f}]")
        
        print(f"\n  CBR (Channel Busy Ratio):")
        print(f"    Average:      {results.get('avg_cbr', 0):.4f}")
        print(f"    Std Dev:      {results.get('std_cbr', 0):.4f}")
        print(f"    CV:           {results.get('cv_cbr', 0):.4f}")
        print(f"    Range:        [{results.get('min_cbr', 0):.4f}, {results.get('max_cbr', 0):.4f}]")
        
        print(f"\n  Neighbors:")
        print(f"    Average:      {results.get('avg_neighbors', 0):.2f}")
        print(f"    Std Dev:      {results.get('std_neighbors', 0):.2f}")
        print(f"    Range:        [{results.get('min_neighbors', 0):.2f}, {results.get('max_neighbors', 0):.2f}]")
        
        print(f"\n  Packets:")
        print(f"    Avg Received: {results.get('avg_packets_received', 0):.0f}")
        print(f"    Avg Sent:     {results.get('avg_packets_sent', 0):.0f}")
        
        print("\n" + "="*80)
        print("Note: CV = Coefficient of Variation (std/mean) - lower is more stable")
        print("="*80)
    
    def save_results(self, results, output_dir="baseline_results"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON summary
        json_file = os.path.join(output_dir, 
                                f"baseline_{self.beaconHz}Hz_{self.txPower}dBm_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] Summary: {json_file}")
        
        # Save detailed CSV
        csv_file = os.path.join(output_dir, 
                               f"baseline_{self.beaconHz}Hz_{self.txPower}dBm_{timestamp}.csv")
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
    
    def generate_plots(self, results, output_dir="baseline_results", show=False):
        """Generate comprehensive visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Baseline Test Results: {self.beaconHz}Hz, {self.txPower}dBm (No RL)', 
                    fontsize=16, fontweight='bold')
        
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
        
        # Row 2: Reward, Neighbors, Cumulative Reward
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
        
        # Row 3: Histograms
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(self.metrics['pdr'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax7.axvline(x=results['avg_pdr'], color='r', linestyle='--', linewidth=2,
                   label=f"Mean: {results['avg_pdr']:.4f}")
        ax7.set_title('PDR Distribution', fontweight='bold')
        ax7.set_xlabel('PDR')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(self.metrics['throughput'], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        ax8.axvline(x=results['avg_throughput'], color='r', linestyle='--', linewidth=2,
                   label=f"Mean: {results['avg_throughput']:.0f}")
        ax8.set_title('Throughput Distribution', fontweight='bold')
        ax8.set_xlabel('Throughput (bps)')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.hist(self.metrics['cbr'], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
        ax9.axvline(x=results['avg_cbr'], color='r', linestyle='--', linewidth=2,
                   label=f"Mean: {results['avg_cbr']:.4f}")
        ax9.set_title('CBR Distribution', fontweight='bold')
        ax9.set_xlabel('CBR')
        ax9.set_ylabel('Frequency')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Save plot
        plot_file = os.path.join(output_dir, 
                                f"baseline_{self.beaconHz}Hz_{self.txPower}dBm_{timestamp}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Plots: {plot_file}")
        
        # Log to WandB if enabled
        if self.use_wandb and self.wandb_run:
            wandb.log({"baseline_results_plot": wandb.Image(plot_file)})
        
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
        description='Baseline Test (No RL Agent) with WandB Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 8 Hz beacon and 23 dBm power
  python test_baseline.py --beaconHz 8 --txPower 23
  
  # Test with WandB logging and custom project
  python test_baseline.py --beaconHz 10 --txPower 26 --wandb --wandb_project "my-vanet-project"
  
  # Full test with all features
  python test_baseline.py --beaconHz 6 --txPower 21 \\
      --wandb --wandb_project "vanet-experiments" \\
      --save --plot --max_steps 1000
  
  # Compare multiple configurations
  python test_baseline.py --beaconHz 4 --txPower 15 --wandb --save
  python test_baseline.py --beaconHz 8 --txPower 23 --wandb --save
  python test_baseline.py --beaconHz 12 --txPower 30 --wandb --save
        """
    )
    
    parser.add_argument('--beaconHz', type=float, required=True,
                       help='Fixed beacon frequency in Hz (e.g., 8)')
    parser.add_argument('--txPower', type=float, required=True,
                       help='Fixed transmission power in dBm (e.g., 23)')
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
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                       help='Output directory for results (default: baseline_results)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='WandB project name (if not specified, will be prompted or use default)')
    parser.add_argument('--wandb_tags', type=str, nargs='+',
                       help='Additional WandB tags')
    parser.add_argument('--run_name', type=str,
                       help='Custom name for this run')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check if WandB is requested but not available
    if args.wandb and not WANDB_AVAILABLE:
        print("[ERROR] WandB requested but not installed. Run: pip install wandb")
        print("[INFO] Continuing without WandB...")
        args.wandb = False
    
    # Set default WandB project if not specified
    if args.wandb and args.wandb_project is None:
        args.wandb_project = 'vanet-baseline-testing'
        print(f"[INFO] Using default WandB project: {args.wandb_project}")
    
    # Validate parameters
    if args.beaconHz <= 0 or args.beaconHz > 20:
        print(f"[WARNING] BeaconHz {args.beaconHz} is outside typical range (0-20 Hz)")
    
    if args.txPower < 10 or args.txPower > 30:
        print(f"[WARNING] TxPower {args.txPower} is outside typical range (10-30 dBm)")
    
    # Create tester
    tester = BaselineTester(
        beaconHz=args.beaconHz,
        txPower=args.txPower,
        port=args.port,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        run_name=args.run_name
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
            "summary/total_reward": results['total_reward'],
            "summary/avg_reward": results['avg_reward'],
            "summary/avg_pdr": results['avg_pdr'],
            "summary/std_pdr": results.get('std_pdr', 0),
            "summary/cv_pdr": results.get('cv_pdr', 0),
            "summary/avg_throughput": results['avg_throughput'],
            "summary/std_throughput": results.get('std_throughput', 0),
            "summary/cv_throughput": results.get('cv_throughput', 0),
            "summary/avg_cbr": results['avg_cbr'],
            "summary/std_cbr": results.get('std_cbr', 0),
            "summary/cv_cbr": results.get('cv_cbr', 0),
            "summary/avg_neighbors": results.get('avg_neighbors', 0),
        })
    
    # Cleanup
    tester.cleanup()
    
    print("\n" + "="*80)
    print("BASELINE TEST COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()

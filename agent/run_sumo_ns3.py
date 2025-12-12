
import os
import sys
import time
import zmq
import json
import traci
import subprocess
import threading
import argparse
import torch

# Import the PPO Agent
# Ensure the agent directory is in the path if needed, but since we are in the same dir it should work
# if running from project root, we might need to adjust sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ppo_dual_continuous import DualControlPPOAgent

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SUMO_CONFIG = os.path.join(BASE_DIR, "../sumo_scenario/highway.sumocfg")
SUMO_CONFIG = os.path.join(BASE_DIR, "../2025-12-12-09-33-54/osm.sumocfg")
NS3_DIR = "/home/tharindu/tarballs/ns-allinone-3.44/ns-3.44"
ZMQ_PORT = 5555
MODEL_PATH = os.path.join(BASE_DIR, "models/ppo_dual_final_20251116_194131.pth")

def run_ns3():
    """Run NS-3 simulation in a separate thread"""
    # Run the binary directly to avoid "program not found" errors with the wrapper
    binary_path = f"{NS3_DIR}/build/scratch/vanet-sumo/vanet-sumo"
    cmd = f"{binary_path}"
    print(f"[Orchestrator] Starting NS-3: {cmd}")
    
    # Enable NS-3 Logging
    env = os.environ.copy()
    env["NS_LOG"] = "VanetSumo=level_all|prefix_time"
    
    subprocess.run(cmd, shell=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    args = parser.parse_args()

    # 0. Initialize PPO Agent (Inference Mode)
    print(f"[Orchestrator] Loading PPO Agent from {MODEL_PATH}...")
    agent = DualControlPPOAgent()
    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
        print("[Orchestrator] Model loaded successfully!")
    else:
        print(f"[Orchestrator] WARNING: Model not found at {MODEL_PATH}. Using random initialization.")

    # 1. Start SUMO
    sumo_cmd = ["sumo-gui" if args.gui else "sumo", "-c", SUMO_CONFIG, "--start"]
    traci.start(sumo_cmd)
    print("[Orchestrator] SUMO started")

    # 2. Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Orchestrator] ZMQ Server bound to port {ZMQ_PORT}")

    # 3. Start NS-3
    ns3_thread = threading.Thread(target=run_ns3)
    ns3_thread.start()

    step = 0
    try:
        while True:
            # Wait for request from NS-3
            msg = socket.recv()
            try:
                data = json.loads(msg.decode())
                msg_type = data.get("type")
            except:
                msg_type = "unknown"
                print(f"Received raw: {msg}")

            if msg_type == "get_positions":
                # Step SUMO
                traci.simulationStep()
                
                # Get positions
                positions = {}
                
                # Maintain a persistent map of SUMO ID -> NS3 ID
                if not hasattr(main, "veh_map"):
                    main.veh_map = {}
                    main.next_id = 0
                
                active_ids = traci.vehicle.getIDList()
                
                # Update map for new vehicles
                for veh_id in active_ids:
                    if veh_id not in main.veh_map:
                        if main.next_id < 100: # Limit to MAX_NODES in NS-3
                            main.veh_map[veh_id] = main.next_id
                            main.next_id += 1
                
                # Build position dict
                for veh_id in active_ids:
                    if veh_id in main.veh_map:
                        x, y = traci.vehicle.getPosition(veh_id)
                        ns3_id = main.veh_map[veh_id]
                        positions[str(ns3_id)] = [x, y]
                
                # Debug: Print active vehicle count
                if step % 100 == 0:
                    print(f"[Orchestrator] Step {step}: SUMO Vehicles={len(active_ids)}, Mapped to NS-3={len(positions)}")

                response = {
                    "type": "positions",
                    "positions": positions
                }
                socket.send_string(json.dumps(response))
                step += 1

            elif msg_type == "state":
                state_data = data.get("data", {})
                
                # Normalize state for agent
                normalized_state = agent.normalize_state(state_data)
                
                # Select Action using PPO Agent (Inference Mode: training=False)
                action_idx, _, _ = agent.select_action(normalized_state, training=False)
                action_params = agent.get_action_params(action_idx)
                
                print(f"[Step {step}] State: PDR={state_data.get('PDR',0):.2f} | "
                      f"Action: Beacon={action_params['beaconHz']}Hz, Tx={action_params['txPower']}dBm")

                # Send Action to NS-3
                response = {"action": action_params}
                socket.send_string(json.dumps(response))

            elif msg_type == "reward":
                reward = data.get("reward", 0.0)
                # print(f"Reward: {reward}")
                # In inference mode, we don't update the agent
                socket.send_string("ack")

            else:
                print(f"Unknown message: {msg}")
                socket.send_string("ack")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        traci.close()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()

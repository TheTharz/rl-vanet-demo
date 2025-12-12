
import os
import sys
import time
import zmq
import json
import traci
import subprocess
import threading
import argparse

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SUMO_CONFIG = os.path.join(BASE_DIR, "../sumo_scenario/highway.sumocfg")
SUMO_CONFIG = os.path.join(BASE_DIR, "../2025-12-12-09-33-54/osm.sumocfg")
NS3_DIR = "/home/tharindu/tarballs/ns-allinone-3.44/ns-3.44"
ZMQ_PORT = 5555

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

    # 1. Start SUMO
    sumo_cmd = ["sumo-gui" if args.gui else "sumo", "-c", SUMO_CONFIG, "--start"]
    traci.start(sumo_cmd)
    print("[Orchestrator] SUMO started")

    # 2. Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Orchestrator] ZMQ Server bound to port {ZMQ_PORT}")

    # 3. Start NS-3 (Manual start required by user for now, or automated)
    # For this demo, we assume user copies files and runs NS-3 manually or we automate it.
    # Let's automate it assuming files are in place.
    ns3_thread = threading.Thread(target=run_ns3)
    ns3_thread.start()

    step = 0
    try:
        while True:
            # Wait for request from NS-3
            # NS-3 sends:
            # 1. "get_positions" -> We step SUMO, get positions, send back
            # 2. "state" -> We receive metrics, send "ack" (or action)
            # 3. "reward" -> We receive reward, send "ack"
            
            msg = socket.recv()
            try:
                data = json.loads(msg.decode())
                msg_type = data.get("type")
            except:
                # Raw string check if JSON fails (legacy)
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
                
                # Collect positions for mapped vehicles
                for veh_id in active_ids:
                    if veh_id in main.veh_map:
                        ns3_id = str(main.veh_map[veh_id])
                        x, y = traci.vehicle.getPosition(veh_id)
                        positions[ns3_id] = [x, y]
                
                response = {"positions": positions}
                
                # Send back
                socket.send_string(json.dumps(response))
                step += 1

            elif msg_type == "state":
                # print(f"State: {data['data']}")
                # Send Action (Dummy for now)
                action = {"action": {"beaconHz": 10, "txPower": 20}}
                socket.send_string(json.dumps(action))

            elif msg_type == "reward":
                # print(f"Reward: {data['reward']}")
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

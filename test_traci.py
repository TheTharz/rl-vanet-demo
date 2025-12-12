import traci
import os
import sys

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CONFIG = os.path.join(BASE_DIR, "sumo_scenario/highway.sumocfg")

def main():
    print(f"Starting SUMO with config: {SUMO_CONFIG}")
    sumo_cmd = ["sumo-gui", "-c", SUMO_CONFIG, "--start", "--quit-on-end"] # GUI test with auto-start
    
    try:
        traci.start(sumo_cmd)
        print("TraCI Connected!")
        
        for i in range(5):
            traci.simulationStep()
            print(f"Step {i}")
            
        traci.close()
        print("TraCI Closed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

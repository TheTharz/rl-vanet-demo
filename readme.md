run the python agent
then run the ns3 script using
./ns3 build
./build/scratch/v2x/scratch_v2x_v2x --lanes=6 --vehPerLane=300 --simTime=10 --beaconHz=10 --payloadBytes=1200 --txPowerDbm=23 --rtsCts=1 --flowmon=1
used ns3 version 3.40

//rl setup

run python script

python dqn_simple_continuous.py --train --max_steps 0 --save_interval 100

python -u dqn_simple_continuous.py --train --max_steps 0 --save_interval 100 > training.log 2>&1

python dqn_simple_continuous.py --train --max_steps 500 --wandb

python dqn_dual_continuous.py --train --max_steps 0 --wandb
python ppo_dual_continuous.py --train --max_steps 0 --wandb
run the vanet

./build/scratch/v2x/scratch_v2x_v2x --vehicles=10 --simTime=7200 --logInterval=5.0 --beaconHz=8.0 --enableRL=1

wandb api key - 6aaafa884d009b3db03e871cd8232cc090d152dd

running the vanet sumo
./build/scratch/v2x_sumo/scratch_v2x_sumo_vanet \
  --simTime=36000 \
  --logInterval=5.0 \ 
  --beaconHz=2.0 \
  --enableRL=1

running the updated vanet with the new pdr calculation

python baseline.py \
  --project "vanet-ppo-dual-control" \
  --name "baseline_60vehicles_8hz_23dbm"

python baseline.py \
  --project "vanet-dqn-both-txpower-beaconhz" \
  --name "baseline_30vehicles_8hz_23dbm"
  
python test_model.py --model models/ppo_dual_final_20251112_054517.pth --wandb --wandb_project vanet-ppo-dual-control

python test_baseline.py --beaconHz 10 --txPower 23 --wandb --wandb_project "rl agent training and testing" --max_steps 10000
//this is here for testing without rl agent

python test_model.py --model models/ppo_dual_final_20251116_194131.pth --wandb --wandb_project "exhibition" --max_steps 10000

python test_model.py --model models/dqn_dual_final_20251116_192041.pth --wandb --wandb_project "exhibition" --max_steps 10000
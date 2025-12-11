# VANET RL Exhibition Dashboard

A real-time web-based dashboard for demonstrating VANET (Vehicular Ad-hoc Network) Reinforcement Learning control systems at exhibitions.

## Features

- 🚀 **Real-time Monitoring**: Live metrics streaming via WebSocket
- 📊 **Interactive Charts**: Beautiful visualizations using Chart.js
- 🔄 **Auto-cycling**: Automatically cycles through PPO, DQN, and Baseline simulations
- 📈 **Performance Comparison**: Side-by-side comparison of all models
- 🎨 **Modern UI**: Dark-themed, responsive dashboard perfect for exhibitions
- 🔌 **Live Updates**: Real-time PDR, Throughput, CBR, BeaconHz, and TX Power monitoring

## System Architecture

```
┌─────────────────┐      WebSocket      ┌──────────────────┐
│   Web Browser   │ ←─────────────────→ │  Dashboard Server│
│  (Dashboard UI) │                     │   (Python/WS)    │
└─────────────────┘                     └──────────────────┘
                                               ↓
                                        ┌──────────────────┐
                                        │  Simulation Mgr  │
                                        │  (test_model.py) │
                                        └──────────────────┘
                                               ↓ ZMQ
                                        ┌──────────────────┐
                                        │  NS3 Simulation  │
                                        │  (VANET Scenario)│
                                        └──────────────────┘
```

## Prerequisites

- Python 3.8+
- NS3 3.40 (built and configured)
- Virtual environment (recommended)
- Trained RL models (PPO and DQN)

## Quick Start

### 1. Setup

Make the startup script executable:

```bash
chmod +x start_dashboard.sh
```

### 2. Start Dashboard

```bash
./start_dashboard.sh
```

This will:
- Activate the virtual environment
- Install all dependencies
- Check for required models
- Start the dashboard server on port 8080

### 3. Access Dashboard

Open your browser and navigate to:
```
http://localhost:8080
```

### 4. Start NS3 Simulation (in a separate terminal)

Start your NS3 simulation with RL enabled:

```bash
# In the NS3 directory
cd /home/tharindu/tarballs/ns-allinone-3.44/ns-3.44
./build/scratch/v2x/scratch_v2x_v2x \
  --vehicles=30 \
  --simTime=600 \
  --enableRL=1
```

NS3 will wait for connection from the Python agent.

### 5. Select and Run Simulation

In the web dashboard:
1. Click on any simulation card (PPO Agent, DQN Agent, or Baseline)
2. The Python agent will start and connect to NS3
3. Watch real-time metrics update on the charts
4. When complete, click another simulation to try a different one

**Note:** Auto-cycling is disabled by default. Each simulation runs once and waits for you to select the next one manually.

## Manual Setup

If you prefer to set up manually:

```bash
# 1. Activate virtual environment
source ../venv/bin/activate

# 2. Install dashboard dependencies
cd dashboard
pip install -r requirements.txt

# 3. Install agent dependencies
cd ../agent
pip install -r requirements.txt

# 4. Start dashboard server
cd ../dashboard
python server.py
```

## Configuration

### Simulation Configurations

Edit `server.py` to customize simulation parameters:

```python
self.simulation_configs = [
    SimulationConfig(
        name="PPO Agent",
        type="ppo",
        model_path="path/to/ppo_model.pth",
        max_steps=500,
        color="#2ecc71"
    ),
    # Add more configurations...
]
```

### Server Port

Change the default port (8080):

```bash
python server.py --port 8000
```

### Disable Auto-cycling

```bash
python server.py --no-auto-cycle
```

## Dashboard Features

### Real-time Metrics

- **PDR (Packet Delivery Ratio)**: Percentage of successfully delivered packets
- **Throughput**: Network throughput in Mbps
- **CBR (Channel Busy Ratio)**: Channel utilization percentage
- **BeaconHz**: Adaptive beacon frequency (controlled by RL agent)
- **TX Power**: Adaptive transmission power (controlled by RL agent)

### Interactive Controls

- **Next Simulation**: Manually cycle to the next simulation
- **Stop**: Stop the current simulation
- **Auto-cycle Toggle**: Enable/disable automatic cycling

### Performance Comparison

The comparison chart shows average performance across all completed simulations, making it easy to see improvements from RL agents over baseline.

## File Structure

```
dashboard/
├── server.py                 # Main server with WebSocket support
├── requirements.txt          # Python dependencies
├── start_dashboard.sh        # Quick start script
├── README.md                # This file
└── static/
    ├── index.html           # Main dashboard HTML
    ├── css/
    │   └── styles.css       # Dashboard styling
    └── js/
        ├── charts.js        # Chart.js visualizations
        └── dashboard.js     # WebSocket client & UI controller
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill the process or use a different port
python server.py --port 8090
```

### WebSocket Connection Failed

- Ensure the server is running
- Check browser console for errors
- Verify firewall settings allow connections to port 8080

### NS3 Simulation Not Connecting

- Ensure NS3 simulation is using `--enableRL=1`
- Check ZMQ port (default: 5555)
- Verify agent is running before starting NS3

### Models Not Found

Ensure trained models exist in the expected locations:
```
agent/models/ppo_dual_final_20251116_194131.pth
agent/models/dqn_dual_final_20251116_192041.pth
```

## Exhibition Tips

### Display Setup

**Recommended: Single Large Display**
- Full HD (1920x1080) or larger
- Browser in fullscreen mode (F11)
- Zoom to 100-125% for better visibility

**Alternative: Multiple Displays**
- Display 1: Dashboard (this)
- Display 2: Terminal with logs
- Display 3: WandB dashboard

### Best Practices

1. **Pre-test Everything**: Run through all simulations before the exhibition
2. **Stable Network**: Use wired connection, not WiFi
3. **Backup Plan**: Have pre-recorded video ready
4. **Quick Restart**: Keep terminals ready to restart components
5. **Monitor Logs**: Watch for errors in server logs

### Performance Optimization

For smoother performance during exhibition:

```python
# In server.py, reduce max_steps for faster cycling
SimulationConfig(
    name="PPO Agent",
    type="ppo",
    model_path="...",
    max_steps=300,  # Reduced from 500
    color="#2ecc71"
)
```

## API Endpoints

### HTTP REST API

- `GET /` - Dashboard UI
- `GET /api/status` - Current status
- `GET /api/configs` - Available configurations
- `POST /api/start` - Start simulation
- `POST /api/stop` - Stop simulation
- `POST /api/cycle` - Cycle to next simulation

### WebSocket Events

**Client → Server:**
- `start_simulation` - Start a specific simulation
- `stop_simulation` - Stop current simulation
- `toggle_auto_cycle` - Enable/disable auto-cycling

**Server → Client:**
- `status` - Current system status
- `simulation_started` - Simulation started event
- `simulation_completed` - Simulation completed event
- `metrics_update` - Real-time metrics update
- `error` - Error notification

## Development

### Running in Development Mode

```bash
# Install dev dependencies
pip install aiohttp-devtools

# Run with auto-reload
adev runserver server.py
```

### Adding New Metrics

1. Update `handleMetricsUpdate()` in `dashboard.js`
2. Add new chart in `charts.js`
3. Add UI element in `index.html`
4. Style in `styles.css`

## License

This dashboard is part of the VANET RL project.

## Support

For issues or questions:
- Check the troubleshooting section
- Review server logs
- Check browser console for client-side errors

## Credits

- **NS3**: Network Simulator 3
- **Chart.js**: Beautiful charts
- **Socket.IO**: Real-time communication
- **aiohttp**: Async web framework

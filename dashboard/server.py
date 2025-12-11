"""
VANET RL Dashboard - WebSocket Server
======================================

Real-time dashboard for monitoring VANET RL simulations.
Supports PPO, DQN agents and baseline testing.

Features:
- Real-time metrics streaming via WebSocket
- Simulation lifecycle management
- Auto-cycling through models
- Performance comparison
"""

import asyncio
import json
import subprocess
import os
import sys
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import queue

# Web framework
from aiohttp import web
import aiohttp_cors
import socketio

# Paths configuration
NS3_DIR = Path("/home/tharindu/tarballs/ns-allinone-3.44/ns-3.44")
AGENT_DIR = Path(__file__).parent.parent / "agent"
sys.path.append(str(AGENT_DIR))

# Import simulation managers
from test_model import ModelTester
from test_baseline import BaselineTester


@dataclass
class SimulationConfig:
    """Configuration for a simulation run"""
    name: str
    type: str  # 'ppo', 'dqn', 'baseline'
    model_path: Optional[str] = None
    beaconHz: Optional[float] = None
    txPower: Optional[float] = None
    max_steps: int = 500
    color: str = "#3498db"


class DashboardServer:
    """Main dashboard server with WebSocket support"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.sio = socketio.AsyncServer(
            async_mode='aiohttp',
            cors_allowed_origins='*',
            ping_timeout=60,
            ping_interval=25
        )
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Simulation state
        self.current_simulation: Optional[SimulationConfig] = None
        self.simulation_process: Optional[subprocess.Popen] = None
        self.current_thread = None
        self.current_tester = None
        self.stop_requested = False
        self.is_running = False
        self.current_step = 0
        self.metrics_queue = queue.Queue()
        self.auto_cycle = True
        
        # Configuration for cycling
        self.simulation_configs = [
            SimulationConfig(
                name="PPO Agent",
                type="ppo",
                model_path=str(AGENT_DIR / "models" / "ppo_dual_final_20251116_194131.pth"),
                max_steps=500,
                color="#2ecc71"
            ),
            SimulationConfig(
                name="DQN Agent",
                type="dqn",
                model_path=str(AGENT_DIR / "models" / "dqn_dual_final_20251116_192041.pth"),
                max_steps=500,
                color="#3498db"
            ),
            SimulationConfig(
                name="Baseline (No RL)",
                type="baseline",
                beaconHz=8.0,
                txPower=23.0,
                max_steps=500,
                color="#e74c3c"
            ),
        ]
        self.current_config_index = 0
        self.auto_cycle = False  # Disabled by default for manual control
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        # Statistics
        self.all_results = []
        
    def _setup_routes(self):
        """Setup HTTP routes"""
        # Serve static files
        dashboard_dir = Path(__file__).parent
        self.app.router.add_static('/static', dashboard_dir / 'static', name='static')
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/api/status', self.status_handler)
        self.app.router.add_get('/api/configs', self.configs_handler)
        self.app.router.add_post('/api/start', self.start_handler)
        self.app.router.add_post('/api/stop', self.stop_handler)
        self.app.router.add_post('/api/cycle', self.cycle_handler)
        
        # Enable CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Add CORS to routes, but skip socket.io routes
        for route in list(self.app.router.routes()):
            if not isinstance(route.resource, web.StaticResource):
                # Skip socket.io routes as they handle their own CORS
                if not str(route.resource).startswith('/socket.io'):
                    try:
                        cors.add(route)
                    except ValueError:
                        # Route already has CORS configured
                        pass
    
    def _setup_socketio(self):
        """Setup SocketIO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            print(f"[WebSocket] Client connected: {sid}")
            # Send current status
            await self.sio.emit('status', {
                'running': self.is_running,
                'current_simulation': asdict(self.current_simulation) if self.current_simulation else None,
                'step': self.current_step
            }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            print(f"[WebSocket] Client disconnected: {sid}")
        
        @self.sio.event
        async def start_simulation(sid, data):
            config_index = data.get('config_index', 0)
            await self.start_simulation(config_index)
        
        @self.sio.event
        async def stop_simulation(sid):
            await self.stop_simulation()
        
        @self.sio.event
        async def toggle_auto_cycle(sid, data):
            self.auto_cycle = data.get('enabled', True)
            await self.sio.emit('auto_cycle_status', {'enabled': self.auto_cycle})
    
    async def index_handler(self, request):
        """Serve index.html"""
        dashboard_dir = Path(__file__).parent
        return web.FileResponse(dashboard_dir / 'static' / 'index.html')
    
    async def status_handler(self, request):
        """Get current status"""
        return web.json_response({
            'running': self.is_running,
            'current_simulation': asdict(self.current_simulation) if self.current_simulation else None,
            'step': self.current_step,
            'auto_cycle': self.auto_cycle,
            'total_results': len(self.all_results)
        })
    
    async def configs_handler(self, request):
        """Get available simulation configurations"""
        return web.json_response({
            'configs': [asdict(c) for c in self.simulation_configs],
            'current_index': self.current_config_index
        })
    
    async def start_handler(self, request):
        """Start a simulation"""
        data = await request.json()
        config_index = data.get('config_index', 0)
        await self.start_simulation(config_index)
        return web.json_response({'success': True})
    
    async def stop_handler(self, request):
        """Stop current simulation"""
        await self.stop_simulation()
        return web.json_response({'success': True})
    
    async def cycle_handler(self, request):
        """Cycle to next simulation"""
        await self.cycle_to_next()
        return web.json_response({'success': True})
    
    async def start_simulation(self, config_index: int):
        """Start a simulation with given config"""
        if self.is_running:
            await self.stop_simulation()
        
        config = self.simulation_configs[config_index]
        self.current_simulation = config
        self.current_config_index = config_index
        self.current_step = 0
        self.is_running = True
        
        print(f"\n[INFO] Starting simulation: {config.name}")
        
        # Emit start event
        await self.sio.emit('simulation_started', {
            'config': asdict(config),
            'timestamp': datetime.now().isoformat()
        })
        
        # Run simulation in background thread
        loop = asyncio.get_event_loop()
        loop.create_task(self._run_simulation_async(config))
    
    async def _run_simulation_async(self, config: SimulationConfig):
        """Run simulation in async context"""
        self.current_thread = None
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Store thread reference for stopping
            def run_with_thread_ref():
                self.current_thread = threading.current_thread()
                self._run_simulation_thread(config)
            
            await loop.run_in_executor(None, run_with_thread_ref)
        except Exception as e:
            print(f"[ERROR] Simulation failed: {e}")
            await self.sio.emit('error', {'message': str(e)})
        finally:
            self.is_running = False
            self.current_thread = None
            await self.sio.emit('simulation_completed', {
                'config': asdict(config),
                'timestamp': datetime.now().isoformat()
            })
            
            # Note: Auto-cycle disabled by default for exhibition
            # Users should manually select simulations from the UI
    
    def _run_simulation_thread(self, config: SimulationConfig):
        """Run simulation in separate thread"""
        try:
            if config.type == 'baseline':
                self._run_baseline(config)
            else:
                self._run_model(config)
        except Exception as e:
            print(f"[ERROR] Simulation thread error: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_baseline(self, config: SimulationConfig):
        """Run baseline simulation"""
        # Force CPU usage for exhibition stability
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        tester = None
        try:
            tester = BaselineTester(
                beaconHz=config.beaconHz,
                txPower=config.txPower,
                port=5555,
                use_wandb=False
            )
            self.current_tester = tester
            
            # Monkey patch to emit metrics on each step
            original_run_test = tester.run_test
            def run_test_with_emit(*args, **kwargs):
                kwargs['verbose'] = True
                last_len = 0
                
                # Run in separate thread and monitor
                import threading
                result = None
                
                def run_original():
                    nonlocal result
                    result = original_run_test(*args, **kwargs)
                
                thread = threading.Thread(target=run_original)
                thread.start()
                
                # Monitor with faster polling
                while thread.is_alive():
                    thread.join(timeout=0.05)  # Poll every 50ms
                    current_len = len(tester.step_data)
                    if current_len > last_len:
                        # Emit all new data points
                        for i in range(last_len, current_len):
                            data = tester.step_data[i]
                            asyncio.run_coroutine_threadsafe(
                                self.sio.emit('metrics_update', {
                                    'step': data['step'],
                                    'pdr': data['pdr'],
                                    'throughput': data['throughput'],
                                    'cbr': data['cbr'],
                                    'beaconHz': data['beaconHz'],
                                    'txPower': data['txPower'],
                                    'config_name': config.name,
                                    'config_type': config.type
                                }),
                                asyncio.get_event_loop()
                            )
                            self.current_step = data['step']
                        last_len = current_len
                
                # Emit any remaining data
                for i in range(last_len, len(tester.step_data)):
                    data = tester.step_data[i]
                    asyncio.run_coroutine_threadsafe(
                        self.sio.emit('metrics_update', {
                            'step': data['step'],
                            'pdr': data['pdr'],
                            'throughput': data['throughput'],
                            'cbr': data['cbr'],
                            'beaconHz': data['beaconHz'],
                            'txPower': data['txPower'],
                            'config_name': config.name,
                            'config_type': config.type
                        }),
                        asyncio.get_event_loop()
                    )
                
                return result
            
            tester.run_test = run_test_with_emit
            
            # Run test
            results = tester.run_test(max_steps=config.max_steps)
            self.all_results.append({
                'config': asdict(config),
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
        finally:
            if tester:
                try:
                    tester.cleanup()
                    print("[INFO] Baseline tester cleaned up")
                except Exception as e:
                    print(f"[WARNING] Cleanup error: {e}")
            self.current_tester = None
    
    def _run_model(self, config: SimulationConfig):
        """Run RL model simulation"""
        # Force CPU usage for exhibition stability
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        tester = None
        try:
            tester = ModelTester(
                model_path=config.model_path,
                model_type='auto',
                port=5555,
                use_wandb=False
            )
            self.current_tester = tester
            
            # Monkey patch to emit metrics on each step
            original_run_test = tester.run_test
            def run_test_with_emit(*args, **kwargs):
                kwargs['verbose'] = True
                last_len = 0
                
                # Run in separate thread and monitor
                import threading
                result = None
                
                def run_original():
                    nonlocal result
                    result = original_run_test(*args, **kwargs)
                
                thread = threading.Thread(target=run_original)
                thread.start()
                
                # Monitor with faster polling
                while thread.is_alive():
                    thread.join(timeout=0.05)  # Poll every 50ms
                    current_len = len(tester.step_data)
                    if current_len > last_len:
                        # Emit all new data points
                        for i in range(last_len, current_len):
                            data = tester.step_data[i]
                            asyncio.run_coroutine_threadsafe(
                                self.sio.emit('metrics_update', {
                                    'step': data['step'],
                                    'pdr': data['pdr'],
                                    'throughput': data['throughput'],
                                    'cbr': data['cbr'],
                                    'beaconHz': data['beaconHz'],
                                    'txPower': data['txPower'],
                                    'config_name': config.name,
                                    'config_type': config.type
                                }),
                                asyncio.get_event_loop()
                            )
                            self.current_step = data['step']
                        last_len = current_len
                
                # Emit any remaining data
                for i in range(last_len, len(tester.step_data)):
                    data = tester.step_data[i]
                    asyncio.run_coroutine_threadsafe(
                        self.sio.emit('metrics_update', {
                            'step': data['step'],
                            'pdr': data['pdr'],
                            'throughput': data['throughput'],
                            'cbr': data['cbr'],
                            'beaconHz': data['beaconHz'],
                            'txPower': data['txPower'],
                            'config_name': config.name,
                            'config_type': config.type
                        }),
                        asyncio.get_event_loop()
                    )
                
                return result
            
            tester.run_test = run_test_with_emit
            
            # Run test
            results = tester.run_test(max_steps=config.max_steps)
            self.all_results.append({
                'config': asdict(config),
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
        finally:
            if tester:
                try:
                    tester.cleanup()
                    print("[INFO] Model tester cleaned up")
                except Exception as e:
                    print(f"[WARNING] Cleanup error: {e}")
            self.current_tester = None
    
    async def stop_simulation(self):
        """Stop current simulation"""
        if not self.is_running:
            return
        
        print("[INFO] Stopping simulation...")
        self.stop_requested = True
        self.is_running = False
        
        # Clean up tester (closes ZMQ socket)
        if self.current_tester:
            try:
                print("[INFO] Cleaning up tester...")
                self.current_tester.cleanup()
                self.current_tester = None
                print("[INFO] Tester cleanup complete")
            except Exception as e:
                print(f"[ERROR] Failed to cleanup tester: {e}")
        
        # Clean up subprocess if exists
        if self.simulation_process:
            try:
                self.simulation_process.terminate()
                self.simulation_process.wait(timeout=5)
            except:
                self.simulation_process.kill()
            self.simulation_process = None
        
        # Reset flags
        self.stop_requested = False
        
        await self.sio.emit('simulation_stopped', {
            'timestamp': datetime.now().isoformat()
        })
    
    async def cycle_to_next(self):
        """Cycle to next simulation configuration"""
        self.current_config_index = (self.current_config_index + 1) % len(self.simulation_configs)
        await self.start_simulation(self.current_config_index)
    
    def run(self):
        """Start the dashboard server"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║     VANET RL Exhibition Dashboard                         ║
╠═══════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:{self.port}              ║
║  WebSocket: ws://localhost:{self.port}/socket.io              ║
║                                                           ║
║  Available simulations:                                   ║
""")
        for i, config in enumerate(self.simulation_configs):
            print(f"║    {i+1}. {config.name:<48}║")
        print(f"""║                                                           ║
║  Click a simulation in the web UI to start               ║
║  Press Ctrl+C to stop                                     ║
╚═══════════════════════════════════════════════════════════╝
""")
        
        print(f"\n✨ Open http://localhost:{self.port} and click on a simulation to start!\n")
        
        web.run_app(self.app, host='0.0.0.0', port=self.port, print=None)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VANET RL Exhibition Dashboard')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--no-auto-cycle', action='store_true', help='Disable auto-cycling')
    
    args = parser.parse_args()
    
    server = DashboardServer(port=args.port)
    server.auto_cycle = not args.no_auto_cycle
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        sys.exit(0)


if __name__ == '__main__':
    main()

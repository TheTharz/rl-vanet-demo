/**
 * VANET RL Dashboard - Main Dashboard Controller
 * Handles WebSocket connections and UI updates
 */

class Dashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.currentSimulation = null;
        this.simulationConfigs = [];
        this.comparisonData = {};

        this.initWebSocket();
        this.initUI();
        this.loadConfigs();
    }

    initWebSocket() {
        console.log('Connecting to WebSocket server...');

        this.socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 10
        });

        // Connection events
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateConnectionStatus(false);
        });

        // Simulation events
        this.socket.on('status', (data) => {
            this.handleStatus(data);
        });

        this.socket.on('simulation_started', (data) => {
            this.handleSimulationStarted(data);
        });

        this.socket.on('simulation_completed', (data) => {
            this.handleSimulationCompleted(data);
        });

        this.socket.on('simulation_stopped', (data) => {
            this.handleSimulationStopped(data);
        });

        this.socket.on('metrics_update', (data) => {
            this.handleMetricsUpdate(data);
        });

        this.socket.on('auto_cycle_status', (data) => {
            document.getElementById('toggle-auto-cycle').checked = data.enabled;
        });

        this.socket.on('error', (data) => {
            console.error('Server error:', data);
            this.showNotification('Error: ' + data.message, 'error');
        });
    }

    initUI() {
        // Cycle button
        document.getElementById('btn-cycle').addEventListener('click', () => {
            this.socket.emit('stop_simulation');
            setTimeout(() => {
                fetch('/api/cycle', { method: 'POST' });
            }, 1000);
        });

        // Stop button
        document.getElementById('btn-stop').addEventListener('click', () => {
            this.socket.emit('stop_simulation');
        });

        // Auto-cycle toggle
        document.getElementById('toggle-auto-cycle').addEventListener('change', (e) => {
            this.socket.emit('toggle_auto_cycle', { enabled: e.target.checked });
        });
    }

    async loadConfigs() {
        try {
            const response = await fetch('/api/configs');
            const data = await response.json();
            this.simulationConfigs = data.configs;
            this.renderQueueList();
        } catch (error) {
            console.error('Failed to load configs:', error);
        }
    }

    renderQueueList() {
        const queueList = document.getElementById('queue-list');
        queueList.innerHTML = '';

        this.simulationConfigs.forEach((config, index) => {
            const item = document.createElement('div');
            item.className = `queue-item ${config.type}`;
            item.innerHTML = `
                <div class="queue-item-name">${config.name}</div>
                <div class="queue-item-type">${config.type.toUpperCase()}</div>
            `;

            item.addEventListener('click', () => {
                this.socket.emit('start_simulation', { config_index: index });
            });

            queueList.appendChild(item);
        });
    }

    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const statusBadge = document.getElementById('connection-status');

        if (connected) {
            statusBadge.classList.remove('disconnected');
            statusBadge.classList.add('connected');
            statusBadge.querySelector('.status-text').textContent = 'Connected';
        } else {
            statusBadge.classList.remove('connected');
            statusBadge.classList.add('disconnected');
            statusBadge.querySelector('.status-text').textContent = 'Disconnected';
        }
    }

    handleStatus(data) {
        console.log('Status:', data);

        if (data.current_simulation) {
            this.currentSimulation = data.current_simulation;
            this.updateSimulationInfo(data.current_simulation, data.step);
        }

        this.updateButtons(data.running);
    }

    handleSimulationStarted(data) {
        console.log('Simulation started:', data.config.name);
        this.currentSimulation = data.config;
        this.updateSimulationInfo(data.config, 0);
        this.updateButtons(true);

        // Reset charts
        window.chartsManager.resetAllCharts();

        // Highlight active queue item
        this.highlightActiveQueueItem(data.config.name);

        this.showNotification(`Started: ${data.config.name}`, 'success');
    }

    handleSimulationCompleted(data) {
        console.log('Simulation completed:', data.config.name);
        this.updateButtons(false);
        this.showNotification(`Completed: ${data.config.name}`, 'success');
    }

    handleSimulationStopped(data) {
        console.log('Simulation stopped');
        this.updateButtons(false);
        this.showNotification('Simulation stopped', 'info');
    }

    handleMetricsUpdate(data) {
        // Update real-time charts
        window.chartsManager.updateMetric('pdr', data.step, data.pdr);
        window.chartsManager.updateMetric('throughput', data.step, data.throughput);
        window.chartsManager.updateMetric('cbr', data.step, data.cbr);
        window.chartsManager.updateMetric('beaconhz', data.step, data.beaconHz);
        window.chartsManager.updateMetric('txpower', data.step, data.txPower);

        // Update quick stats
        this.updateQuickStats({
            pdr: data.pdr,
            throughput: data.throughput,
            cbr: data.cbr,
            beaconHz: data.beaconHz,
            txPower: data.txPower
        });

        // Update progress
        if (this.currentSimulation) {
            this.updateProgress(data.step, this.currentSimulation.max_steps);
        }

        // Store for comparison
        if (!this.comparisonData[data.config_name]) {
            this.comparisonData[data.config_name] = {
                name: data.config_name,
                type: data.config_type,
                pdr: [],
                throughput: [],
                cbr: []
            };
        }

        this.comparisonData[data.config_name].pdr.push(data.pdr);
        this.comparisonData[data.config_name].throughput.push(data.throughput);
        this.comparisonData[data.config_name].cbr.push(data.cbr);

        // Update comparison chart with averages
        this.updateComparisonChart();
    }

    updateSimulationInfo(config, step) {
        document.getElementById('current-sim-name').textContent = config.name;
        document.getElementById('current-sim-step').textContent =
            `Step: ${step} / ${config.max_steps}`;

        // Update color theme
        const simName = document.getElementById('current-sim-name');
        simName.style.color = config.color;
    }

    updateProgress(current, total) {
        const percentage = (current / total) * 100;
        const progressBar = document.getElementById('progress-bar');
        progressBar.style.width = percentage + '%';
    }

    updateQuickStats(stats) {
        document.getElementById('stat-pdr').textContent =
            (stats.pdr * 100).toFixed(2) + '%';
        document.getElementById('stat-throughput').textContent =
            stats.throughput.toFixed(3) + ' Mbps';
        document.getElementById('stat-cbr').textContent =
            (stats.cbr * 100).toFixed(2) + '%';
        document.getElementById('stat-beaconhz').textContent =
            stats.beaconHz.toFixed(1) + ' Hz';
        document.getElementById('stat-txpower').textContent =
            stats.txPower.toFixed(1) + ' dBm';
    }

    updateButtons(running) {
        document.getElementById('btn-cycle').disabled = !running;
        document.getElementById('btn-stop').disabled = !running;
    }

    highlightActiveQueueItem(name) {
        const items = document.querySelectorAll('.queue-item');
        items.forEach(item => {
            if (item.querySelector('.queue-item-name').textContent === name) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    updateComparisonChart() {
        const comparisonData = Object.values(this.comparisonData).map(sim => {
            const avgPdr = sim.pdr.reduce((a, b) => a + b, 0) / sim.pdr.length;
            const avgThroughput = sim.throughput.reduce((a, b) => a + b, 0) / sim.throughput.length;
            const avgCbr = sim.cbr.reduce((a, b) => a + b, 0) / sim.cbr.length;

            return {
                name: sim.name,
                type: sim.type,
                pdr: avgPdr,
                throughput: avgThroughput,
                cbr: avgCbr
            };
        });

        if (comparisonData.length > 0) {
            window.chartsManager.updateComparison(comparisonData);
        }
    }

    showNotification(message, type = 'info') {
        // Simple console notification for now
        // You can implement a toast notification system here
        console.log(`[${type.toUpperCase()}] ${message}`);

        // Flash the header briefly
        const header = document.querySelector('.header');
        header.style.transition = 'opacity 0.3s';
        header.style.opacity = '0.8';
        setTimeout(() => {
            header.style.opacity = '1';
        }, 300);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing VANET RL Dashboard...');
    window.dashboard = new Dashboard();
});

/**
 * VANET RL Dashboard - Charts Manager
 * Handles all Chart.js visualizations
 */

class ChartsManager {
    constructor() {
        this.charts = {};
        this.maxDataPoints = 100;
        this.colors = {
            ppo: '#2ecc71',
            dqn: '#3498db',
            baseline: '#e74c3c'
        };
        
        this.initCharts();
    }
    
    initCharts() {
        // Common chart options
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                duration: 500
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#bdc3c7',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(22, 33, 62, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#bdc3c7',
                    borderColor: '#34495e',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(52, 73, 94, 0.3)'
                    },
                    ticks: {
                        color: '#bdc3c7',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(52, 73, 94, 0.3)'
                    },
                    ticks: {
                        color: '#bdc3c7'
                    }
                }
            }
        };
        
        // PDR Chart
        this.charts.pdr = new Chart(document.getElementById('chart-pdr'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PDR',
                    data: [],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: {
                        ...commonOptions.scales.y,
                        min: 0,
                        max: 1,
                        ticks: {
                            ...commonOptions.scales.y.ticks,
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
        
        // Throughput Chart
        this.charts.throughput = new Chart(document.getElementById('chart-throughput'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Throughput (Mbps)',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: commonOptions
        });
        
        // CBR Chart
        this.charts.cbr = new Chart(document.getElementById('chart-cbr'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CBR',
                    data: [],
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: {
                        ...commonOptions.scales.y,
                        min: 0,
                        max: 1
                    }
                }
            }
        });
        
        // BeaconHz Chart
        this.charts.beaconhz = new Chart(document.getElementById('chart-beaconhz'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Beacon Frequency (Hz)',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    stepped: true
                }]
            },
            options: commonOptions
        });
        
        // TX Power Chart
        this.charts.txpower = new Chart(document.getElementById('chart-txpower'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'TX Power (dBm)',
                    data: [],
                    borderColor: '#e67e22',
                    backgroundColor: 'rgba(230, 126, 34, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    stepped: true
                }]
            },
            options: commonOptions
        });
        
        // Comparison Charts (Bar)
        const comparisonOptions = {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#bdc3c7',
                        font: {
                            size: 14
                        }
                    }
                }
            },
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    beginAtZero: true
                }
            }
        };
        
        this.charts.comparisonPdr = new Chart(document.getElementById('chart-comparison-pdr'), {
            type: 'bar',
            data: {
                labels: ['PPO Agent', 'DQN Agent', 'Baseline (No RL)'],
                datasets: [{
                    label: 'PDR',
                    data: [0, 0, 0],
                    backgroundColor: ['#2ecc71', '#3498db', '#e74c3c'],
                    borderColor: ['#27ae60', '#2980b9', '#c0392b'],
                    borderWidth: 2
                }]
            },
            options: {
                ...comparisonOptions,
                scales: {
                    ...comparisonOptions.scales,
                    y: {
                        ...comparisonOptions.scales.y,
                        min: 0,
                        max: 1,
                        ticks: {
                            ...comparisonOptions.scales.y.ticks,
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
        
        this.charts.comparisonThroughput = new Chart(document.getElementById('chart-comparison-throughput'), {
            type: 'bar',
            data: {
                labels: ['PPO Agent', 'DQN Agent', 'Baseline (No RL)'],
                datasets: [{
                    label: 'Throughput (bps)',
                    data: [0, 0, 0],
                    backgroundColor: ['#2ecc71', '#3498db', '#e74c3c'],
                    borderColor: ['#27ae60', '#2980b9', '#c0392b'],
                    borderWidth: 2
                }]
            },
            options: comparisonOptions
        });
        
        this.charts.comparisonCbr = new Chart(document.getElementById('chart-comparison-cbr'), {
            type: 'bar',
            data: {
                labels: ['PPO Agent', 'DQN Agent', 'Baseline (No RL)'],
                datasets: [{
                    label: 'CBR (inverted - higher is better)',
                    data: [0, 0, 0],
                    backgroundColor: ['#2ecc71', '#3498db', '#e74c3c'],
                    borderColor: ['#27ae60', '#2980b9', '#c0392b'],
                    borderWidth: 2
                }]
            },
            options: {
                ...comparisonOptions,
                scales: {
                    ...comparisonOptions.scales,
                    y: {
                        ...comparisonOptions.scales.y,
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }
    
    updateMetric(metric, step, value) {
        const chart = this.charts[metric];
        if (!chart) return;
        
        // Add new data point
        chart.data.labels.push(step);
        chart.data.datasets[0].data.push(value);
        
        // Keep only last N points
        if (chart.data.labels.length > this.maxDataPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none'); // Update without animation for better performance
    }
    
    resetChart(metric) {
        const chart = this.charts[metric];
        if (!chart) return;
        
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.update();
    }
    
    resetAllCharts() {
        Object.keys(this.charts).forEach(key => {
            if (!key.startsWith('comparison')) {
                this.resetChart(key);
            }
        });
    }
    
    updateComparison(data) {
        // data = [{name: 'PPO', type: 'ppo', pdr: 0.95, throughput: 5.2, cbr: 0.3}, ...]
        // Update each comparison chart separately
        const ppoData = data.find(d => d.type === 'ppo') || {};
        const dqnData = data.find(d => d.type === 'dqn') || {};
        const baselineData = data.find(d => d.type === 'baseline') || {};
        
        // Update PDR comparison
        this.charts.comparisonPdr.data.datasets[0].data = [
            ppoData.pdr || 0,
            dqnData.pdr || 0,
            baselineData.pdr || 0
        ];
        this.charts.comparisonPdr.update();
        
        // Update Throughput comparison
        this.charts.comparisonThroughput.data.datasets[0].data = [
            ppoData.throughput || 0,
            dqnData.throughput || 0,
            baselineData.throughput || 0
        ];
        this.charts.comparisonThroughput.update();
        
        // Update CBR comparison (inverted - lower CBR is better)
        this.charts.comparisonCbr.data.datasets[0].data = [
            1 - (ppoData.cbr || 0),
            1 - (dqnData.cbr || 0),
            1 - (baselineData.cbr || 0)
        ];
        this.charts.comparisonCbr.update();
    }
    
    addComparisonData(name, type, metrics) {
        // Update comparison charts with data from completed simulation
        const index = type === 'ppo' ? 0 : type === 'dqn' ? 1 : 2;
        
        // Update PDR
        this.charts.comparisonPdr.data.datasets[0].data[index] = metrics.pdr || 0;
        this.charts.comparisonPdr.update();
        
        // Update Throughput
        this.charts.comparisonThroughput.data.datasets[0].data[index] = metrics.throughput || 0;
        this.charts.comparisonThroughput.update();
        
        // Update CBR (inverted)
        this.charts.comparisonCbr.data.datasets[0].data[index] = 1 - (metrics.cbr || 0);
        this.charts.comparisonCbr.update();
    }
}

// Global charts instance
window.chartsManager = new ChartsManager();

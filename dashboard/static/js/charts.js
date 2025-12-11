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
        
        // Comparison Chart (Bar)
        this.charts.comparison = new Chart(document.getElementById('chart-comparison'), {
            type: 'bar',
            data: {
                labels: ['PDR', 'Throughput', 'CBR (inverted)'],
                datasets: []
            },
            options: {
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
            if (key !== 'comparison') {
                this.resetChart(key);
            }
        });
    }
    
    updateComparison(data) {
        // data = [{name: 'PPO', type: 'ppo', pdr: 0.95, throughput: 5.2, cbr: 0.3}, ...]
        const chart = this.charts.comparison;
        
        chart.data.datasets = data.map(sim => ({
            label: sim.name,
            data: [
                sim.pdr || 0,
                sim.throughput || 0,
                (1 - (sim.cbr || 0)) // Invert CBR (lower is better)
            ],
            backgroundColor: this.colors[sim.type] || '#95a5a6',
            borderColor: this.colors[sim.type] || '#7f8c8d',
            borderWidth: 2
        }));
        
        chart.update();
    }
    
    addComparisonData(name, type, metrics) {
        // Add or update a dataset in comparison chart
        const chart = this.charts.comparison;
        const existingIndex = chart.data.datasets.findIndex(ds => ds.label === name);
        
        const dataset = {
            label: name,
            data: [
                metrics.pdr || 0,
                metrics.throughput || 0,
                (1 - (metrics.cbr || 0))
            ],
            backgroundColor: this.colors[type] || '#95a5a6',
            borderColor: this.colors[type] || '#7f8c8d',
            borderWidth: 2
        };
        
        if (existingIndex >= 0) {
            chart.data.datasets[existingIndex] = dataset;
        } else {
            chart.data.datasets.push(dataset);
        }
        
        chart.update();
    }
}

// Global charts instance
window.chartsManager = new ChartsManager();

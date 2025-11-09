// Adaptive Learning Monitor JavaScript

const API_BASE = window.location.origin;
let refreshInterval = null;
let activityLog = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Adaptive Learning Monitor initialized');
    
    // Setup event listeners
    document.getElementById('refreshBtn').addEventListener('click', refreshAllData);
    document.getElementById('retrainBtn').addEventListener('click', triggerRetrain);
    document.getElementById('rebalanceBtn').addEventListener('click', triggerRebalance);
    
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    
    // Symbol/Model change
    document.getElementById('monitorSymbol').addEventListener('change', refreshAllData);
    document.getElementById('monitorModel').addEventListener('change', refreshAllData);
    
    // Initial load
    refreshAllData();
    
    // Auto-refresh every 10 seconds for real-time monitoring
    refreshInterval = setInterval(() => {
        const activeTab = document.querySelector('.tab.active').dataset.tab;
        refreshAllData();
        // Reload active tab data
        if (activeTab !== 'overview') {
            setTimeout(() => loadTabData(activeTab), 500);
        }
    }, 10000);
    
    addActivity('info', 'Monitor initialized - Real-time updates every 10s');
});

// Switch tabs
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');
    
    // Load tab-specific data
    loadTabData(tabName);
}

// Load data for specific tab
function loadTabData(tabName) {
    const symbol = document.getElementById('monitorSymbol').value;
    const model = document.getElementById('monitorModel').value;
    
    switch(tabName) {
        case 'performance':
            loadPerformanceTrend(symbol, model);
            break;
        case 'versions':
            loadVersionHistory(symbol, model);
            break;
        case 'ensemble':
            loadEnsembleData(symbol);
            break;
        case 'errors':
            loadErrorAnalysis(symbol, model);
            break;
        case 'logs':
            loadTrainingLogs(symbol, model);
            break;
    }
}

// Refresh all data
async function refreshAllData() {
    const symbol = document.getElementById('monitorSymbol').value;
    const model = document.getElementById('monitorModel').value;
    
    addActivity('info', `Refreshing data for ${symbol}/${model}`);
    
    try {
        await Promise.all([
            loadSchedulerStatus(),
            loadModelStatistics(symbol, model),
            loadTabData(document.querySelector('.tab.active').dataset.tab)
        ]);
        
        addActivity('success', 'Data refreshed successfully');
    } catch (error) {
        addActivity('error', `Refresh failed: ${error.message}`);
    }
}

// Load scheduler status
async function loadSchedulerStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/scheduler/status`);
        const data = await response.json();
        
        if (data.success) {
            const status = data.status;
            document.getElementById('schedulerStatus').textContent = 
                status.is_running ? '🟢 Running' : '🔴 Stopped';
            document.getElementById('monitoredSymbols').textContent = 
                status.monitored_symbols.join(', ') || 'None';
            document.getElementById('nextRun').textContent = 
                status.next_run || 'N/A';
        }
    } catch (error) {
        console.error('Error loading scheduler status:', error);
    }
}

// Load model statistics
async function loadModelStatistics(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/${symbol}/${model}`);
        const data = await response.json();
        
        if (data.success && data.statistics.status !== 'no_data') {
            const stats = data.statistics;
            
            // Update overview metrics
            document.getElementById('totalPredictions').textContent = 
                stats.total_predictions || 0;
            document.getElementById('trainingCount').textContent = 
                stats.training_count || 0;
            document.getElementById('daysSinceTraining').textContent = 
                stats.days_since_training !== null ? `${stats.days_since_training} days` : 'N/A';
            
            // Performance metrics
            if (stats.recent_performance && stats.recent_performance.mape !== null) {
                document.getElementById('recentMAPE').textContent = 
                    `${stats.recent_performance.mape.toFixed(2)}%`;
            }
            
            if (stats.baseline_performance && stats.baseline_performance.mape !== null) {
                document.getElementById('baselineMAPE').textContent = 
                    `${stats.baseline_performance.mape.toFixed(2)}%`;
            }
            
            if (stats.all_time_performance && stats.all_time_performance.mape !== null) {
                document.getElementById('alltimeMAPE').textContent = 
                    `${stats.all_time_performance.mape.toFixed(2)}%`;
            }
            
            // Performance status
            const recentMAPE = stats.recent_performance?.mape || 0;
            const baselineMAPE = stats.baseline_performance?.mape || 0;
            
            let statusText = '✅ Good';
            let statusColor = '#00ff41';
            
            if (recentMAPE > baselineMAPE * 1.2) {
                statusText = '⚠️ Degraded';
                statusColor = '#ffaa00';
            } else if (recentMAPE > baselineMAPE * 1.5) {
                statusText = '❌ Poor';
                statusColor = '#ff0040';
            }
            
            const statusEl = document.getElementById('performanceStatus');
            statusEl.textContent = statusText;
            statusEl.style.color = statusColor;
        } else {
            // No data available
            document.getElementById('totalPredictions').textContent = '0';
            document.getElementById('trainingCount').textContent = '0';
            document.getElementById('daysSinceTraining').textContent = 'N/A';
            document.getElementById('recentMAPE').textContent = 'N/A';
            document.getElementById('baselineMAPE').textContent = 'N/A';
            document.getElementById('alltimeMAPE').textContent = 'N/A';
            document.getElementById('performanceStatus').textContent = 'No Data';
        }
    } catch (error) {
        console.error('Error loading model statistics:', error);
    }
}

// Load performance trend
async function loadPerformanceTrend(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/trend/${symbol}/${model}?days=30`);
        const data = await response.json();
        
        if (data.success && data.trend.length > 0) {
            const trend = data.trend;
            
            const trace = {
                x: trend.map(t => t.date),
                y: trend.map(t => t.mape),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MAPE',
                line: {
                    color: '#00ff41',
                    width: 2
                },
                marker: {
                    size: 6,
                    color: '#00ff41'
                }
            };
            
            const layout = {
                title: {
                    text: `${model.toUpperCase()} Performance Trend`,
                    font: { color: '#00ff41' }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
                xaxis: {
                    title: 'Date',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                yaxis: {
                    title: 'MAPE (%)',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                font: { family: 'Inter, sans-serif', color: '#ffffff' }
            };
            
            Plotly.newPlot('performanceTrendChart', [trace], layout, {responsive: true});
        } else {
            document.getElementById('performanceTrendChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No performance data available</p>';
        }
    } catch (error) {
        console.error('Error loading performance trend:', error);
    }
}

// Load version history
async function loadVersionHistory(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/versions/${symbol}/${model}?limit=10`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const historyHTML = data.history.map(v => {
                const isActive = v.status === 'active';
                const badgeClass = isActive ? 'version-badge active' : 'version-badge';
                const date = new Date(v.trained_at).toLocaleString();
                const mape = v.performance?.mape?.toFixed(2) || 'N/A';
                
                return `
                    <div class="log-entry">
                        <span class="${badgeClass}">${v.version}</span>
                        ${isActive ? '<span class="version-badge active">ACTIVE</span>' : ''}
                        <div class="log-time">${date}</div>
                        <div class="metric-row">
                            <span class="metric-label">MAPE:</span>
                            <span class="metric-value">${mape}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Update Type:</span>
                            <span class="metric-value">${v.update_type || 'N/A'}</span>
                        </div>
                    </div>
                `;
            }).join('');
            
            document.getElementById('versionHistory').innerHTML = historyHTML;
            document.getElementById('activeVersion').textContent = 
                data.history.find(v => v.status === 'active')?.version || 'N/A';
        } else {
            document.getElementById('versionHistory').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5);">No version history available</p>';
            document.getElementById('activeVersion').textContent = 'N/A';
        }
    } catch (error) {
        console.error('Error loading version history:', error);
    }
}

// Load ensemble data
async function loadEnsembleData(symbol) {
    try {
        // Load current weights
        const weightsResponse = await fetch(`${API_BASE}/api/adaptive/ensemble/weights/${symbol}`);
        const weightsData = await weightsResponse.json();
        
        if (weightsData.success && weightsData.weights) {
            const weights = weightsData.weights;
            
            const weightsHTML = Object.entries(weights)
                .sort((a, b) => b[1] - a[1])
                .map(([model, weight]) => {
                    const percentage = (weight * 100).toFixed(1);
                    return `
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>${model.toUpperCase()}</span>
                                <span style="color: #00ff41; font-weight: 600;">${percentage}%</span>
                            </div>
                            <div class="weight-bar" style="width: ${percentage}%;">
                                ${percentage}%
                            </div>
                        </div>
                    `;
                }).join('');
            
            document.getElementById('ensembleWeights').innerHTML = weightsHTML;
        } else {
            document.getElementById('ensembleWeights').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5);">No ensemble weights available</p>';
        }
        
        // Load weight history
        const historyResponse = await fetch(`${API_BASE}/api/adaptive/ensemble/history/${symbol}?days=30`);
        const historyData = await historyResponse.json();
        
        if (historyData.success && historyData.history.length > 0) {
            const history = historyData.history;
            
            // Extract unique model names
            const modelNames = [...new Set(history.flatMap(h => Object.keys(h.weights)))];
            
            // Create traces for each model
            const traces = modelNames.map(modelName => ({
                x: history.map(h => h.timestamp),
                y: history.map(h => (h.weights[modelName] || 0) * 100),
                type: 'scatter',
                mode: 'lines',
                name: modelName.toUpperCase(),
                line: { width: 2 }
            }));
            
            const layout = {
                title: {
                    text: 'Ensemble Weight Evolution',
                    font: { color: '#00ff41' }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
                xaxis: {
                    title: 'Date',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                yaxis: {
                    title: 'Weight (%)',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)',
                    range: [0, 100]
                },
                font: { family: 'Inter, sans-serif', color: '#ffffff' }
            };
            
            Plotly.newPlot('weightHistoryChart', traces, layout, {responsive: true});
        } else {
            document.getElementById('weightHistoryChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No weight history available</p>';
        }
    } catch (error) {
        console.error('Error loading ensemble data:', error);
    }
}

// Load training logs
async function loadTrainingLogs(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/training/logs/${symbol}/${model}?limit=20`);
        const data = await response.json();
        
        if (data.success && data.logs.length > 0) {
            const logsHTML = data.logs.map(log => {
                const statusClass = log.status === 'success' ? '' : 
                                  log.status === 'failed' ? 'error' : 'warning';
                const startTime = new Date(log.training_started).toLocaleString();
                const duration = log.training_completed ? 
                    Math.round((new Date(log.training_completed) - new Date(log.training_started)) / 1000) : 
                    'N/A';
                
                return `
                    <div class="log-entry ${statusClass}">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <strong>${log.version}</strong>
                            <span class="version-badge">${log.trigger}</span>
                        </div>
                        <div class="log-time">${startTime}</div>
                        <div class="metric-row">
                            <span class="metric-label">Status:</span>
                            <span class="metric-value">${log.status}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Duration:</span>
                            <span class="metric-value">${duration}s</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Epochs:</span>
                            <span class="metric-value">${log.epochs}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Final MAPE:</span>
                            <span class="metric-value">${log.metrics?.mape?.toFixed(2) || 'N/A'}%</span>
                        </div>
                    </div>
                `;
            }).join('');
            
            document.getElementById('trainingLogs').innerHTML = logsHTML;
        } else {
            document.getElementById('trainingLogs').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5);">No training logs available</p>';
        }
    } catch (error) {
        console.error('Error loading training logs:', error);
    }
}

// Trigger manual retrain
async function triggerRetrain() {
    const symbol = document.getElementById('monitorSymbol').value;
    const model = document.getElementById('monitorModel').value;
    
    if (!confirm(`Trigger retraining for ${symbol}/${model}?`)) {
        return;
    }
    
    addActivity('info', `Triggering retrain for ${symbol}/${model}...`);
    
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/retrain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, model })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addActivity('success', `Retraining started for ${symbol}/${model}`);
            // Refresh data after a delay
            setTimeout(refreshAllData, 5000);
        } else {
            addActivity('error', `Retrain failed: ${data.error}`);
        }
    } catch (error) {
        addActivity('error', `Retrain error: ${error.message}`);
    }
}

// Trigger ensemble rebalance
async function triggerRebalance() {
    const symbol = document.getElementById('monitorSymbol').value;
    
    if (!confirm(`Rebalance ensemble weights for ${symbol}?`)) {
        return;
    }
    
    addActivity('info', `Rebalancing ensemble for ${symbol}...`);
    
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/ensemble/rebalance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, lookback_days: 7 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addActivity('success', `Ensemble rebalanced for ${symbol}`);
            // Refresh ensemble data
            loadEnsembleData(symbol);
        } else {
            addActivity('error', `Rebalance failed: ${data.error}`);
        }
    } catch (error) {
        addActivity('error', `Rebalance error: ${error.message}`);
    }
}

// Add activity to feed
function addActivity(type, message) {
    const timestamp = new Date().toLocaleTimeString();
    const typeClass = type === 'error' ? 'error' : type === 'warning' ? 'warning' : '';
    const icon = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️';
    
    const entry = `
        <div class="log-entry ${typeClass}">
            <span>${icon} ${message}</span>
            <div class="log-time">${timestamp}</div>
        </div>
    `;
    
    const feed = document.getElementById('activityFeed');
    feed.insertAdjacentHTML('afterbegin', entry);
    
    // Keep only last 50 entries
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
    
    // Add to log array
    activityLog.unshift({ type, message, timestamp });
    if (activityLog.length > 100) {
        activityLog.pop();
    }
}

// Load error analysis
async function loadErrorAnalysis(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/prediction-errors/${symbol}/${model}?days=30`);
        const data = await response.json();
        
        if (data.success && data.errors.length > 0) {
            const errors = data.errors;
            
            // Chart 1: Actual vs Predicted with Error Overlay
            const actualTrace = {
                x: errors.map(e => e.date),
                y: errors.map(e => e.actual),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual Price',
                line: { color: '#00ff41', width: 2 },
                marker: { size: 4 }
            };
            
            const predictedTrace = {
                x: errors.map(e => e.date),
                y: errors.map(e => e.predicted),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: { color: '#ffaa00', width: 2, dash: 'dash' },
                marker: { size: 4 }
            };
            
            const errorTrace = {
                x: errors.map(e => e.date),
                y: errors.map(e => e.error),
                type: 'bar',
                name: 'Absolute Error',
                yaxis: 'y2',
                marker: {
                    color: errors.map(e => e.error_percentage > 5 ? '#ff0040' : '#00ff41'),
                    opacity: 0.6
                }
            };
            
            const layout1 = {
                title: {
                    text: `${model.toUpperCase()} - ${symbol}: Actual vs Predicted Prices`,
                    font: { color: '#00ff41', size: 16 }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
                xaxis: {
                    title: 'Date',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                yaxis: {
                    title: 'Price (USD)',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                yaxis2: {
                    title: 'Error',
                    overlaying: 'y',
                    side: 'right',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                font: { family: 'Inter, sans-serif', color: '#ffffff' },
                margin: { t: 50, b: 50, l: 60, r: 60 },
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(0, 0, 0, 0.5)',
                    bordercolor: 'rgba(0, 255, 65, 0.3)',
                    borderwidth: 1
                }
            };
            
            Plotly.newPlot('errorAnalysisChart', [actualTrace, predictedTrace, errorTrace], layout1, {responsive: true});
            
            // Chart 2: Error Distribution (Histogram)
            const errorPercentages = errors.map(e => e.error_percentage);
            
            const histogramTrace = {
                x: errorPercentages,
                type: 'histogram',
                name: 'Error Distribution',
                marker: {
                    color: '#00ff41',
                    line: {
                        color: '#00cc33',
                        width: 1
                    }
                },
                nbinsx: 20
            };
            
            const layout2 = {
                title: {
                    text: 'Prediction Error Distribution',
                    font: { color: '#00ff41', size: 16 }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
                xaxis: {
                    title: 'Error Percentage (%)',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                yaxis: {
                    title: 'Frequency',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                font: { family: 'Inter, sans-serif', color: '#ffffff' },
                margin: { t: 50, b: 50, l: 60, r: 30 }
            };
            
            Plotly.newPlot('errorDistributionChart', [histogramTrace], layout2, {responsive: true});
            
        } else {
            document.getElementById('errorAnalysisChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No error data available. Generate some forecasts first.</p>';
            document.getElementById('errorDistributionChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No error data available.</p>';
        }
    } catch (error) {
        console.error('Error loading error analysis:', error);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});

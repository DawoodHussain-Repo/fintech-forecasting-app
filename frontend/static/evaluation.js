// Model Evaluation JavaScript

const API_BASE = window.location.origin;
let allModels = [];
let selectedModel = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Model Evaluation page initialized');
    
    // Setup event listeners
    document.getElementById('refreshModels').addEventListener('click', loadTrainedModels);
    document.getElementById('filterSymbol').addEventListener('change', filterModels);
    document.getElementById('filterModel').addEventListener('change', filterModels);
    
    // Load trained models
    loadTrainedModels();
});

// Load all trained models from database
async function loadTrainedModels() {
    try {
        addLoadingState();
        
        const response = await fetch(`${API_BASE}/api/adaptive/trained-models`);
        const data = await response.json();
        
        if (data.success && data.models.length > 0) {
            allModels = data.models;
            
            // Populate symbol filter
            const symbols = [...new Set(data.models.map(m => m.symbol))];
            populateSymbolFilter(symbols);
            
            // Display models
            displayModels(allModels);
        } else {
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading trained models:', error);
        showErrorState(error.message);
    }
}

// Populate symbol filter dropdown
function populateSymbolFilter(symbols) {
    const filterSelect = document.getElementById('filterSymbol');
    
    // Keep "All Symbols" option
    filterSelect.innerHTML = '<option value="all">All Symbols</option>';
    
    // Add symbol options
    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol;
        option.textContent = symbol;
        filterSelect.appendChild(option);
    });
}

// Filter models based on selected filters
function filterModels() {
    const symbolFilter = document.getElementById('filterSymbol').value;
    const modelFilter = document.getElementById('filterModel').value;
    
    let filtered = allModels;
    
    if (symbolFilter !== 'all') {
        filtered = filtered.filter(m => m.symbol === symbolFilter);
    }
    
    if (modelFilter !== 'all') {
        filtered = filtered.filter(m => m.model_name === modelFilter);
    }
    
    displayModels(filtered);
}

// Display models in grid
function displayModels(models) {
    const container = document.getElementById('modelList');
    
    if (models.length === 0) {
        showEmptyState();
        return;
    }
    
    const html = models.map(model => {
        const mape = model.recent_mape?.toFixed(2) || 'N/A';
        const predictions = model.total_predictions || 0;
        const versions = model.version_count || 1;
        const daysSince = model.days_since_training !== null ? `${model.days_since_training}d` : 'N/A';
        
        // Performance status
        let statusColor = '#00ff41';
        let statusText = 'Good';
        if (model.recent_mape > 5) {
            statusColor = '#ff0040';
            statusText = 'Poor';
        } else if (model.recent_mape > 3) {
            statusColor = '#ffaa00';
            statusText = 'Fair';
        }
        
        return `
            <div class="model-card" onclick="selectModel('${model.symbol}', '${model.model_name}')">
                <div class="model-header">
                    <div class="model-name">${model.model_name.toUpperCase()}</div>
                    <div class="model-badge">${model.symbol}</div>
                </div>
                <div class="model-stats">
                    <div class="stat-row">
                        <span class="stat-label">Recent MAPE:</span>
                        <span class="stat-value" style="color: ${statusColor}">${mape}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Status:</span>
                        <span class="stat-value" style="color: ${statusColor}">${statusText}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Predictions:</span>
                        <span class="stat-value">${predictions}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Versions:</span>
                        <span class="stat-value">${versions}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Last Trained:</span>
                        <span class="stat-value">${daysSince} ago</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

// Select a model to view details
async function selectModel(symbol, modelName) {
    selectedModel = { symbol, modelName };
    
    // Highlight selected card
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    
    // Show details section
    document.getElementById('modelDetails').style.display = 'block';
    
    // Scroll to details
    document.getElementById('modelDetails').scrollIntoView({ behavior: 'smooth' });
    
    // Load model details
    await Promise.all([
        loadPerformanceOverview(symbol, modelName),
        loadPerformanceTrend(symbol, modelName),
        loadVersionHistory(symbol, modelName),
        loadVersionComparison(symbol, modelName)
    ]);
}

// Load performance overview
async function loadPerformanceOverview(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/${symbol}/${modelName}`);
        const data = await response.json();
        
        if (data.success && data.statistics.status !== 'no_data') {
            const stats = data.statistics;
            
            const recent = stats.recent_performance || {};
            const baseline = stats.baseline_performance || {};
            const allTime = stats.all_time_performance || {};
            
            // Calculate changes
            const mapeChange = recent.mape && baseline.mape ? 
                ((recent.mape - baseline.mape) / baseline.mape * 100).toFixed(1) : null;
            
            const html = `
                <div class="comparison-card">
                    <div class="comparison-label">Recent MAPE</div>
                    <div class="comparison-value">${recent.mape?.toFixed(2) || 'N/A'}%</div>
                    ${mapeChange ? `<div class="comparison-change ${mapeChange > 0 ? 'negative' : 'positive'}">
                        ${mapeChange > 0 ? '↑' : '↓'} ${Math.abs(mapeChange)}% vs baseline
                    </div>` : ''}
                </div>
                <div class="comparison-card">
                    <div class="comparison-label">Baseline MAPE</div>
                    <div class="comparison-value">${baseline.mape?.toFixed(2) || 'N/A'}%</div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-label">All-time MAPE</div>
                    <div class="comparison-value">${allTime.mape?.toFixed(2) || 'N/A'}%</div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-label">Total Predictions</div>
                    <div class="comparison-value">${stats.total_predictions || 0}</div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-label">Training Count</div>
                    <div class="comparison-value">${stats.training_count || 0}</div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-label">Days Since Training</div>
                    <div class="comparison-value">${stats.days_since_training !== null ? stats.days_since_training : 'N/A'}</div>
                </div>
            `;
            
            document.getElementById('performanceOverview').innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading performance overview:', error);
    }
}

// Load performance trend
async function loadPerformanceTrend(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/trend/${symbol}/${modelName}?days=30`);
        const data = await response.json();
        
        if (data.success && data.trend.length > 0) {
            const trend = data.trend;
            
            const trace = {
                x: trend.map(t => t.date),
                y: trend.map(t => t.mape),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MAPE',
                line: { color: '#00ff41', width: 2 },
                marker: { size: 6, color: '#00ff41' }
            };
            
            const layout = {
                title: {
                    text: `${modelName.toUpperCase()} - ${symbol} Performance (30 Days)`,
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
                    title: 'MAPE (%)',
                    gridcolor: 'rgba(0, 255, 65, 0.1)',
                    color: 'rgba(255, 255, 255, 0.7)'
                },
                font: { family: 'Inter, sans-serif', color: '#ffffff' },
                margin: { t: 50, b: 50, l: 60, r: 30 }
            };
            
            Plotly.newPlot('performanceTrendChart', [trace], layout, {responsive: true});
        } else {
            document.getElementById('performanceTrendChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No performance trend data available</p>';
        }
    } catch (error) {
        console.error('Error loading performance trend:', error);
    }
}

// Load version history
async function loadVersionHistory(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/versions/${symbol}/${modelName}?limit=10`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const html = data.history.map((v, index) => {
                const isActive = v.status === 'active';
                const date = new Date(v.trained_at).toLocaleString();
                const mape = v.performance?.mape?.toFixed(2) || 'N/A';
                const rmse = v.performance?.rmse?.toFixed(2) || 'N/A';
                const mae = v.performance?.mae?.toFixed(2) || 'N/A';
                
                // Calculate improvement from previous version
                let improvement = '';
                if (index < data.history.length - 1 && v.performance?.mape && data.history[index + 1].performance?.mape) {
                    const prevMape = data.history[index + 1].performance.mape;
                    const change = ((prevMape - v.performance.mape) / prevMape * 100).toFixed(1);
                    if (change > 0) {
                        improvement = `<div style="color: #00ff41; font-size: 0.85em; margin-top: 5px;">
                            ↑ ${change}% improvement from previous
                        </div>`;
                    } else if (change < 0) {
                        improvement = `<div style="color: #ff0040; font-size: 0.85em; margin-top: 5px;">
                            ↓ ${Math.abs(change)}% degradation from previous
                        </div>`;
                    }
                }
                
                return `
                    <div class="version-item ${isActive ? 'active' : ''}">
                        <div class="version-header">
                            <span class="version-number">${v.version}</span>
                            ${isActive ? '<span class="model-badge">ACTIVE</span>' : ''}
                        </div>
                        <div class="version-date">${date}</div>
                        <div class="model-stats" style="margin-top: 10px;">
                            <div class="stat-row">
                                <span class="stat-label">MAPE:</span>
                                <span class="stat-value">${mape}%</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">RMSE:</span>
                                <span class="stat-value">${rmse}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">MAE:</span>
                                <span class="stat-value">${mae}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Update Type:</span>
                                <span class="stat-value">${v.update_type || 'N/A'}</span>
                            </div>
                        </div>
                        ${improvement}
                    </div>
                `;
            }).join('');
            
            document.getElementById('versionHistory').innerHTML = html;
        } else {
            document.getElementById('versionHistory').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5);">No version history available</p>';
        }
    } catch (error) {
        console.error('Error loading version history:', error);
    }
}

// Load version comparison chart
async function loadVersionComparison(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/versions/${symbol}/${modelName}?limit=10`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const versions = data.history.reverse(); // Oldest to newest
            
            // Create separate traces for each metric
            const mapeTrace = {
                x: versions.map(v => v.version),
                y: versions.map(v => v.performance?.mape || 0),
                type: 'bar',
                name: 'MAPE (%)',
                marker: { 
                    color: '#00ff41',
                    line: {
                        color: '#00cc33',
                        width: 1
                    }
                },
                text: versions.map(v => (v.performance?.mape || 0).toFixed(2) + '%'),
                textposition: 'outside',
                hovertemplate: '<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
            };
            
            const rmseTrace = {
                x: versions.map(v => v.version),
                y: versions.map(v => v.performance?.rmse || 0),
                type: 'bar',
                name: 'RMSE',
                marker: { 
                    color: '#888888',
                    line: {
                        color: '#666666',
                        width: 1
                    }
                },
                text: versions.map(v => (v.performance?.rmse || 0).toFixed(2)),
                textposition: 'outside',
                yaxis: 'y2',
                hovertemplate: '<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>'
            };
            
            const layout = {
                title: {
                    text: 'Version Performance Comparison',
                    font: { color: '#ffffff', size: 16 }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: '#0a0a0a',
                xaxis: {
                    title: 'Version',
                    gridcolor: '#1a1a1a',
                    color: '#888888',
                    tickangle: -45
                },
                yaxis: {
                    title: 'MAPE (%)',
                    gridcolor: '#1a1a1a',
                    color: '#888888'
                },
                yaxis2: {
                    title: 'RMSE',
                    overlaying: 'y',
                    side: 'right',
                    color: '#888888',
                    gridcolor: '#1a1a1a'
                },
                font: { family: 'Inter, sans-serif', color: '#ffffff' },
                margin: { t: 50, b: 80, l: 60, r: 60 },
                barmode: 'group',
                bargap: 0.3,
                bargroupgap: 0.1,
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1.1,
                    orientation: 'h',
                    bgcolor: 'rgba(0, 0, 0, 0)',
                    bordercolor: '#1a1a1a',
                    borderwidth: 1
                }
            };
            
            Plotly.newPlot('versionComparisonChart', [mapeTrace, rmseTrace], layout, {responsive: true});
        } else {
            document.getElementById('versionComparisonChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No version comparison data available</p>';
        }
    } catch (error) {
        console.error('Error loading version comparison:', error);
    }
}

// Show empty state
function showEmptyState() {
    const container = document.getElementById('modelList');
    container.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">📭</div>
            <h3>No Trained Models Found</h3>
            <p>Generate some forecasts first to see model evaluations here</p>
            <a href="/" class="btn-primary" style="display: inline-block; margin-top: 20px; text-decoration: none;">
                Go to Forecast Page
            </a>
        </div>
    `;
}

// Show error state
function showErrorState(message) {
    const container = document.getElementById('modelList');
    container.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">❌</div>
            <h3>Error Loading Models</h3>
            <p>${message}</p>
            <button onclick="loadTrainedModels()" class="btn-primary" style="margin-top: 20px;">
                Try Again
            </button>
        </div>
    `;
}

// Show loading state
function addLoadingState() {
    const container = document.getElementById('modelList');
    container.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">⏳</div>
            <h3>Loading Models...</h3>
            <p>Please wait</p>
        </div>
    `;
}

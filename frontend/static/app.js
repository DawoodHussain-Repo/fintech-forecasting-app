// API base URL
const API_BASE = window.location.origin;

// Event listeners
document.getElementById('forecastBtn').addEventListener('click', generateForecast);
document.getElementById('compareBtn').addEventListener('click', compareModels);
document.getElementById('symbol').addEventListener('change', handleSymbolChange);

// Handle custom symbol input visibility
function handleSymbolChange() {
    const symbolSelect = document.getElementById('symbol');
    const customSymbolGroup = document.getElementById('customSymbolGroup');
    
    if (symbolSelect.value === 'CUSTOM') {
        customSymbolGroup.style.display = 'flex';
        document.getElementById('customSymbol').focus();
    } else {
        customSymbolGroup.style.display = 'none';
    }
}

// Get the actual symbol to use
function getSelectedSymbol() {
    const symbolSelect = document.getElementById('symbol');
    
    if (symbolSelect.value === 'CUSTOM') {
        const customSymbol = document.getElementById('customSymbol').value.trim().toUpperCase();
        if (!customSymbol) {
            alert('Please enter a custom symbol (e.g., TSLA, NFLX, DOGE-USD)');
            return null;
        }
        return customSymbol;
    }
    
    return symbolSelect.value;
}

async function generateForecast() {
    const symbol = getSelectedSymbol();
    if (!symbol) return;
    
    const model = document.getElementById('model').value;
    const horizon = document.getElementById('horizon').value;
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('metrics').style.display = 'none';
    document.getElementById('comparison').style.display = 'none';
    document.getElementById('chart').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol, model, horizon })
        });
        
        if (!response.ok) {
            const text = await response.text();
            console.error('Server response:', text);
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayMetrics(data.metrics, data.latest_data_time, data.prediction_time);
            displayChart(data.historical_data, data.predictions, symbol);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Forecast error:', error);
        alert('Error generating forecast: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

async function compareModels() {
    const symbol = getSelectedSymbol();
    if (!symbol) return;
    
    const horizon = document.getElementById('horizon').value;
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('metrics').style.display = 'none';
    document.getElementById('chart').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/compare_models`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol, horizon })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayComparison(data.results);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error comparing models: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayMetrics(metrics, latestDataTime, predictionTime) {
    document.getElementById('rmse').textContent = metrics.rmse.toFixed(2);
    document.getElementById('mae').textContent = metrics.mae.toFixed(2);
    document.getElementById('mape').textContent = metrics.mape.toFixed(2) + '%';
    document.getElementById('model-name').textContent = metrics.model;
    
    // Add accuracy interpretation
    const mape = metrics.mape;
    let accuracyText = '';
    let accuracyColor = '';
    
    if (mape < 5) {
        accuracyText = 'Excellent';
        accuracyColor = '#00ff41';
    } else if (mape < 10) {
        accuracyText = 'Good';
        accuracyColor = '#00cc33';
    } else if (mape < 20) {
        accuracyText = 'Fair';
        accuracyColor = '#ffaa00';
    } else {
        accuracyText = 'Poor';
        accuracyColor = '#ff0040';
    }
    
    document.getElementById('accuracy-rating').textContent = accuracyText;
    document.getElementById('accuracy-rating').style.color = accuracyColor;
    
    // Show data freshness info
    if (latestDataTime) {
        const dataDate = new Date(latestDataTime);
        const predDate = new Date(predictionTime);
        document.getElementById('data-time').textContent = dataDate.toLocaleString();
        document.getElementById('prediction-time').textContent = predDate.toLocaleString();
    }
    
    document.getElementById('metrics').style.display = 'block';
}

function displayChart(historicalData, predictions, symbol) {
    // Prepare historical data
    const historicalDates = historicalData.map(d => d.date);
    const historicalClose = historicalData.map(d => d.close);
    const historicalOpen = historicalData.map(d => d.open);
    const historicalHigh = historicalData.map(d => d.high);
    const historicalLow = historicalData.map(d => d.low);
    
    // Prepare prediction data
    const predictionDates = predictions.map(p => p.date);
    const predictionValues = predictions.map(p => p.predicted_close);
    
    // Create candlestick trace for historical data
    const candlestickTrace = {
        x: historicalDates,
        close: historicalClose,
        high: historicalHigh,
        low: historicalLow,
        open: historicalOpen,
        type: 'candlestick',
        name: 'Historical',
        increasing: { line: { color: '#00ff41' } },
        decreasing: { line: { color: '#ff0040' } }
    };
    
    // Create line trace for predictions
    const predictionTrace = {
        x: predictionDates,
        y: predictionValues,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Forecast',
        line: {
            color: '#00ff41',
            width: 3,
            dash: 'dash'
        },
        marker: {
            size: 8,
            color: '#00ff41',
            line: {
                color: '#000',
                width: 2
            }
        }
    };
    
    // Create error overlay trace (shows prediction errors)
    // Calculate error bars based on historical prediction accuracy
    const lastActualPrice = historicalClose[historicalClose.length - 1];
    const errorPercentage = 0.02; // 2% error margin (can be dynamic based on MAPE)
    
    const errorBandUpper = predictionValues.map(v => v * (1 + errorPercentage));
    const errorBandLower = predictionValues.map(v => v * (1 - errorPercentage));
    
    const errorBandTrace = {
        x: predictionDates.concat(predictionDates.slice().reverse()),
        y: errorBandUpper.concat(errorBandLower.slice().reverse()),
        fill: 'toself',
        fillcolor: 'rgba(0, 255, 65, 0.1)',
        line: { color: 'transparent' },
        name: 'Error Margin',
        type: 'scatter',
        showlegend: true,
        hoverinfo: 'skip'
    };
    
    // Layout with dark theme
    const layout = {
        title: {
            text: `${symbol} Price Forecast`,
            font: { 
                size: 24,
                color: '#00ff41',
                family: 'Inter, sans-serif'
            }
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
        xaxis: {
            title: {
                text: 'Date',
                font: { color: '#00ff41' }
            },
            gridcolor: 'rgba(0, 255, 65, 0.1)',
            color: 'rgba(255, 255, 255, 0.7)',
            rangeslider: { visible: false }
        },
        yaxis: {
            title: {
                text: 'Price (USD)',
                font: { color: '#00ff41' }
            },
            gridcolor: 'rgba(0, 255, 65, 0.1)',
            color: 'rgba(255, 255, 255, 0.7)'
        },
        hovermode: 'x unified',
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(0, 0, 0, 0.5)',
            bordercolor: 'rgba(0, 255, 65, 0.3)',
            borderwidth: 1,
            font: { color: '#ffffff' }
        },
        font: {
            family: 'Inter, sans-serif',
            color: '#ffffff'
        }
    };
    
    // Plot with dark theme config
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };
    
    Plotly.newPlot('chart', [candlestickTrace, errorBandTrace, predictionTrace], layout, config);
}

function displayComparison(results) {
    const tbody = document.querySelector('#comparisonTable tbody');
    tbody.innerHTML = '';
    
    // Sort by RMSE
    const sortedModels = Object.entries(results).sort((a, b) => 
        a[1].metrics.rmse - b[1].metrics.rmse
    );
    
    sortedModels.forEach(([modelName, data]) => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td><strong>${modelName.toUpperCase()}</strong></td>
            <td>${data.metrics.rmse.toFixed(2)}</td>
            <td>${data.metrics.mae.toFixed(2)}</td>
            <td>${data.metrics.mape.toFixed(2)}%</td>
        `;
    });
    
    document.getElementById('comparison').style.display = 'block';
}

// Initialize on page load
window.addEventListener('load', () => {
    console.log('Stock Forecasting App Loaded');
});

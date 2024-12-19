// Global Variables
let currentSymbol = '';
let chartInstance = null;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Main.js loaded');
    
    // Initialize Select2
    $('#stockSelect').select2({
        placeholder: 'Select a stock...',
        allowClear: true
    });
    
    // Initialize Chart.js
    const ctx = document.getElementById('priceChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Stock Price',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    // Event Listeners
    const stockSelect = document.getElementById('stockSelect');
    const predictionPeriod = document.getElementById('predictionPeriod');
    const addStockButton = document.getElementById('addStockButton');
    
    stockSelect.addEventListener('change', function() {
        const symbol = this.value;
        if (symbol) {
            currentSymbol = symbol;
            updateData();
        }
    });
    
    predictionPeriod.addEventListener('change', function() {
        if (currentSymbol) {
            updateData();
        }
    });
    
    // Handle dark mode toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    darkModeToggle.addEventListener('change', function() {
        document.body.classList.toggle('dark-mode');
        updateChartTheme(this.checked);
    });
});

function showLoading(show = true) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = show ? 'flex' : 'none';
}

function updateChart(historyData) {
    if (!chartInstance) return;
    
    const dates = historyData.map(d => new Date(d.date).toLocaleDateString());
    const prices = historyData.map(d => d.close);
    
    chartInstance.data.labels = dates;
    chartInstance.data.datasets[0].data = prices;
    
    if (historyData.prediction) {
        // Add prediction point
        const lastDate = new Date(dates[dates.length - 1]);
        const predictionDate = new Date(lastDate);
        predictionDate.setDate(predictionDate.getDate() + 1);
        
        chartInstance.data.labels.push(predictionDate.toLocaleDateString());
        chartInstance.data.datasets[0].data.push(historyData.prediction.price);
        
        // Add prediction interval
        chartInstance.data.datasets.push({
            label: 'Prediction Interval',
            data: Array(dates.length).fill(null).concat([historyData.prediction.lower, historyData.prediction.price, historyData.prediction.upper]),
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'transparent',
            fill: true
        });
    }
    
    chartInstance.update();
}

function updatePrices(predictionData) {
    const currentPrice = document.getElementById('currentPrice');
    const predictedPrice = document.getElementById('predictedPrice');
    const priceChange = document.getElementById('priceChange').querySelector('span');
    const predictionConfidence = document.getElementById('predictionConfidence').querySelector('span');
    
    if (predictionData.current) {
        currentPrice.textContent = `$${predictionData.current.toFixed(2)}`;
        const change = ((predictionData.current - predictionData.previous) / predictionData.previous * 100);
        priceChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
        priceChange.className = change >= 0 ? 'price-up' : 'price-down';
    }
    
    if (predictionData.predicted) {
        predictedPrice.textContent = `$${predictionData.predicted.toFixed(2)}`;
        predictionConfidence.textContent = `${(predictionData.confidence * 100).toFixed(1)}%`;
    }
}

function updateTechnicalIndicators(data) {
    const container = document.getElementById('technicalIndicators');
    container.innerHTML = '';
    
    const indicators = [
        { name: 'RSI', value: data.rsi, threshold: { oversold: 30, overbought: 70 } },
        { name: 'MACD', value: data.macd, signal: data.macd_signal },
        { name: 'Stochastic', value: data.stoch_k, signal: data.stoch_d },
        { name: 'Bollinger Bands', value: data.bb_middle, upper: data.bb_upper, lower: data.bb_lower }
    ];
    
    indicators.forEach(indicator => {
        const col = document.createElement('div');
        col.className = 'col-md-3 mb-3';
        
        const card = document.createElement('div');
        card.className = 'card indicator-card';
        
        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';
        
        const title = document.createElement('h6');
        title.textContent = indicator.name;
        
        const value = document.createElement('p');
        value.className = 'mb-0';
        
        if (indicator.threshold) {
            const val = parseFloat(indicator.value);
            value.textContent = val.toFixed(2);
            value.className += val > indicator.threshold.overbought ? ' price-up' : 
                             val < indicator.threshold.oversold ? ' price-down' : '';
        } else if (indicator.signal) {
            value.textContent = `${parseFloat(indicator.value).toFixed(2)} / ${parseFloat(indicator.signal).toFixed(2)}`;
        } else if (indicator.upper && indicator.lower) {
            value.textContent = `${parseFloat(indicator.lower).toFixed(2)} - ${parseFloat(indicator.value).toFixed(2)} - ${parseFloat(indicator.upper).toFixed(2)}`;
        }
        
        cardBody.appendChild(title);
        cardBody.appendChild(value);
        card.appendChild(cardBody);
        col.appendChild(card);
        container.appendChild(col);
    });
}

async function updateData() {
    try {
        showLoading(true);
        
        const period = document.getElementById('predictionPeriod').value;
        const response = await fetch(`/api/stock/${currentSymbol}?period=${period}`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch stock data');
        }
        
        const data = await response.json();
        
        // Update chart
        updateChart(data.history);
        
        // Update prices
        updatePrices({
            current: data.current_price,
            previous: data.previous_close,
            predicted: data.prediction.price,
            confidence: data.prediction.confidence
        });
        
        // Update technical indicators
        updateTechnicalIndicators(data.indicators);
        
    } catch (error) {
        console.error('Error updating data:', error);
        alert('Failed to fetch stock data. Please try again.');
    } finally {
        showLoading(false);
    }
}

function updateChartTheme(isDark) {
    if (!chartInstance) return;
    
    const theme = {
        light: {
            backgroundColor: 'white',
            gridColor: 'rgba(0, 0, 0, 0.1)',
            textColor: 'black'
        },
        dark: {
            backgroundColor: '#333',
            gridColor: 'rgba(255, 255, 255, 0.1)',
            textColor: 'white'
        }
    };
    
    const currentTheme = isDark ? theme.dark : theme.light;
    
    chartInstance.options.plugins.legend.labels.color = currentTheme.textColor;
    chartInstance.options.scales.x.grid.color = currentTheme.gridColor;
    chartInstance.options.scales.y.grid.color = currentTheme.gridColor;
    chartInstance.options.scales.x.ticks.color = currentTheme.textColor;
    chartInstance.options.scales.y.ticks.color = currentTheme.textColor;
    
    chartInstance.update();
}

function getPrediction(symbol) {
    showLoading('prediction');
    fetch(`/api/predict/${symbol}`)
        .then(response => response.json())
        .then(data => {
            hideLoading('prediction');
            if (data.error) {
                showError('prediction', data.error);
                return;
            }
            
            const predictionCard = document.getElementById('prediction-card');
            const currentPrice = parseFloat(data.current_price).toFixed(2);
            const predictedPrice = parseFloat(data.predicted_price).toFixed(2);
            const change = ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
            const direction = change >= 0 ? 'up' : 'down';
            
            predictionCard.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">Price Prediction</h5>
                    <p class="card-text">Current Price: $${currentPrice}</p>
                    <p class="card-text">Predicted Price: $${predictedPrice}</p>
                    <p class="card-text ${direction}">Expected Change: ${change}%</p>
                    <p class="card-text"><small class="text-muted">Prediction Date: ${data.prediction_date}</small></p>
                </div>
            `;
        })
        .catch(error => {
            hideLoading('prediction');
            showError('prediction', 'Failed to get prediction');
            console.error('Error:', error);
        });
}

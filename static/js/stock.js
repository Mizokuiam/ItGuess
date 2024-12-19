// Stock data management and chart rendering
let currentSymbol = null;
let stockChart = null;

async function addStock() {
    const symbolInput = document.getElementById('stockSymbol');
    const symbol = symbolInput.value.trim().toUpperCase();
    const errorDiv = document.getElementById('stockError');
    
    if (!symbol) {
        showError('Please enter a stock symbol');
        return;
    }
    
    try {
        const response = await fetch('/api/add_stock', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol })
        });
        
        const data = await response.json();
        
        if (data.success) {
            symbolInput.value = '';
            errorDiv.classList.add('d-none');
            await loadStockData(symbol);
        } else {
            showError(data.message || 'Failed to add stock');
        }
    } catch (error) {
        showError('Error adding stock: ' + error.message);
    }
}

async function loadStockData(symbol, period = '1m') {
    try {
        const response = await fetch(`/api/stock/${symbol}?period=${period}`);
        const data = await response.json();
        
        if (data.success) {
            currentSymbol = symbol;
            updateChart(data);
            updateIndicators(data.indicators);
            updatePrediction(data.predicted);
        } else {
            showError(data.error || 'Failed to load stock data');
        }
    } catch (error) {
        showError('Error loading stock data: ' + error.message);
    }
}

function updateChart(data) {
    const chartDiv = document.getElementById('stockChart');
    
    const trace = {
        x: data.prices.map(p => p.date),
        y: data.prices.map(p => p.price),
        type: 'scatter',
        mode: 'lines',
        name: currentSymbol,
        line: {
            color: '#2196F3',
            width: 2
        }
    };
    
    const layout = {
        title: `${currentSymbol} Stock Price`,
        xaxis: {
            title: 'Date',
            showgrid: false
        },
        yaxis: {
            title: 'Price ($)',
            showgrid: true
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 30, b: 40, l: 60, r: 30 }
    };
    
    Plotly.newPlot(chartDiv, [trace], layout);
}

function updateIndicators(indicators) {
    const container = document.getElementById('technicalIndicators');
    container.innerHTML = `
        <div class="row">
            <div class="col-4">
                <div class="indicator-card">
                    <h6>RSI</h6>
                    <p class="mb-0 ${getRSIClass(indicators.RSI)}">${indicators.RSI}</p>
                </div>
            </div>
            <div class="col-4">
                <div class="indicator-card">
                    <h6>MACD</h6>
                    <p class="mb-0">${indicators.MACD}</p>
                </div>
            </div>
            <div class="col-4">
                <div class="indicator-card">
                    <h6>Signal</h6>
                    <p class="mb-0">${indicators.Signal}</p>
                </div>
            </div>
        </div>
    `;
}

function updatePrediction(prediction) {
    const container = document.getElementById('predictions');
    if (!container) return;
    
    const priceChange = prediction.predicted_price - prediction.current_price;
    const changePercent = (priceChange / prediction.current_price) * 100;
    const changeClass = priceChange >= 0 ? 'text-success' : 'text-danger';
    
    container.innerHTML = `
        <div class="col-md-4">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Current Price</h6>
                    <p class="h4">$${prediction.current_price.toFixed(2)}</p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Predicted Price</h6>
                    <p class="h4">$${prediction.predicted_price.toFixed(2)}</p>
                    <p class="mb-0 ${changeClass}">
                        ${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%
                    </p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Confidence</h6>
                    <p class="h4">${(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
            </div>
        </div>
    `;
}

function getRSIClass(rsi) {
    if (rsi > 70) return 'text-danger';
    if (rsi < 30) return 'text-success';
    return 'text-warning';
}

function showError(message) {
    const errorDiv = document.getElementById('stockError');
    errorDiv.textContent = message;
    errorDiv.classList.remove('d-none');
}

// Initialize with a default stock
document.addEventListener('DOMContentLoaded', () => {
    loadStockData('AAPL');
});

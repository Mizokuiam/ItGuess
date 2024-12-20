// Global variables
let currentSymbol = '';
let stockChart = null;
let predictionChart = null;

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('stockSearch');
    const searchButton = document.getElementById('searchButton');
    const searchResults = document.getElementById('searchResults');

    // Setup search functionality
    searchInput.addEventListener('input', debounce(handleSearch, 300));
    searchButton.addEventListener('click', () => handleSearch(searchInput.value));
    
    // Handle enter key in search
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSearch(searchInput.value);
        }
    });
});

// Debounce function to limit API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle stock search
async function handleSearch(query) {
    if (!query) return;
    
    try {
        const response = await fetch(`/api/search_stock?query=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        const searchResults = document.getElementById('searchResults');
        searchResults.innerHTML = '';
        
        if (data.results && data.results.length > 0) {
            data.results.forEach(stock => {
                const item = document.createElement('a');
                item.href = '#';
                item.className = 'list-group-item list-group-item-action';
                item.innerHTML = `<strong>${stock.symbol}</strong> - ${stock.name}`;
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    loadStock(stock.symbol);
                    searchResults.innerHTML = '';
                });
                searchResults.appendChild(item);
            });
        } else {
            searchResults.innerHTML = '<div class="list-group-item">No results found</div>';
        }
    } catch (error) {
        console.error('Search error:', error);
        const searchResults = document.getElementById('searchResults');
        searchResults.innerHTML = '<div class="list-group-item text-danger">Error searching for stocks</div>';
    }
}

// Load stock data and update UI
async function loadStock(symbol) {
    try {
        currentSymbol = symbol;
        document.getElementById('stockData').classList.remove('d-none');
        document.getElementById('stockSymbol').textContent = symbol;
        
        const response = await fetch(`/api/stock/${symbol}`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        updatePriceInfo(data.current);
        updateChart(data.prices);
        updateIndicators(data.indicators);
        updatePrediction(data.predicted);
        
    } catch (error) {
        console.error('Error loading stock:', error);
        alert('Error loading stock data. Please try again.');
    }
}

// Update price information
function updatePriceInfo(current) {
    const priceElement = document.getElementById('currentPrice');
    const changeElement = document.getElementById('priceChange');
    
    priceElement.textContent = `$${current.price.toFixed(2)}`;
    const changeText = current.change.toFixed(2);
    const changeClass = current.change >= 0 ? 'text-success' : 'text-danger';
    const changeSymbol = current.change >= 0 ? '▲' : '▼';
    
    changeElement.textContent = `${changeSymbol} ${Math.abs(changeText)}%`;
    changeElement.className = changeClass;
}

// Update stock chart
function updateChart(prices) {
    const dates = prices.map(p => p.date);
    const values = prices.map(p => p.price);
    
    const trace = {
        x: dates,
        y: values,
        type: 'scatter',
        mode: 'lines',
        name: currentSymbol,
        line: {
            color: '#17a2b8',
            width: 2
        }
    };
    
    const layout = {
        title: 'Stock Price History',
        xaxis: {
            title: 'Date',
            rangeslider: { visible: true }
        },
        yaxis: {
            title: 'Price ($)'
        },
        showlegend: true
    };
    
    Plotly.newPlot('stockChart', [trace], layout);
}

// Update technical indicators
function updateIndicators(indicators) {
    document.getElementById('rsiValue').textContent = indicators.RSI;
    document.getElementById('macdValue').textContent = indicators.MACD;
    document.getElementById('signalValue').textContent = indicators.Signal;
}

// Update prediction chart
function updatePrediction(predicted) {
    if (!predicted || !predicted.dates || !predicted.values) return;
    
    const trace = {
        x: predicted.dates,
        y: predicted.values,
        type: 'scatter',
        mode: 'lines',
        name: 'Predicted',
        line: {
            color: '#28a745',
            width: 2,
            dash: 'dot'
        }
    };
    
    const layout = {
        title: 'Price Prediction',
        xaxis: {
            title: 'Date'
        },
        yaxis: {
            title: 'Price ($)'
        },
        showlegend: true
    };
    
    Plotly.newPlot('predictionChart', [trace], layout);
}

// Initialize with a default stock
document.addEventListener('DOMContentLoaded', () => {
    loadStock('AAPL');
});

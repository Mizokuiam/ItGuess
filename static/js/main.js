// Global Variables
let currentSymbol = '';
let chartInstance = null;

// DOM Ready
$(document).ready(function() {
    // Initialize tooltips and popovers
    $('[data-bs-toggle="tooltip"]').tooltip();
    $('[data-bs-toggle="popover"]').popover();
    
    // Setup search form
    $('#stockSearchForm').on('submit', handleStockSearch);
    
    // Setup technical indicator toggles
    $('.indicator-toggle').on('change', handleIndicatorToggle);
    
    // Setup timeframe buttons
    $('.timeframe-btn').on('click', handleTimeframeChange);
    
    // Setup dark mode toggle
    $('#darkModeToggle').on('change', handleDarkModeToggle);
    
    // Load user preferences
    loadUserPreferences();
});

// Stock Search Handler
async function handleStockSearch(e) {
    e.preventDefault();
    const symbol = $('#stockSymbol').val().toUpperCase();
    
    showLoading();
    try {
        const response = await fetch(`/api/stock/${symbol}`);
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        currentSymbol = symbol;
        updateStockInfo(data);
        updateChart(data.prices);
        updateIndicators(data.indicators);
        updatePredictions(data.predictions);
        
    } catch (error) {
        showError('Failed to fetch stock data');
    } finally {
        hideLoading();
    }
}

// Chart Management
function updateChart(prices) {
    const trace = {
        x: prices.map(p => p.date),
        y: prices.map(p => p.close),
        type: 'scatter',
        name: currentSymbol
    };
    
    const layout = {
        title: `${currentSymbol} Price History`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        showlegend: true
    };
    
    Plotly.newPlot('stockChart', [trace], layout);
}

// Technical Indicators
function handleIndicatorToggle(e) {
    const indicator = $(this).data('indicator');
    const isChecked = $(this).prop('checked');
    
    if (isChecked) {
        showLoading();
        fetch(`/api/indicator/${currentSymbol}/${indicator}`)
            .then(response => response.json())
            .then(data => {
                addIndicatorToChart(data, indicator);
            })
            .catch(error => showError('Failed to load indicator'))
            .finally(() => hideLoading());
    } else {
        removeIndicatorFromChart(indicator);
    }
}

function addIndicatorToChart(data, indicator) {
    const trace = {
        x: data.dates,
        y: data.values,
        type: 'scatter',
        name: indicator,
        line: { dash: 'dot' }
    };
    
    Plotly.addTraces('stockChart', trace);
}

function removeIndicatorFromChart(indicator) {
    const chart = document.getElementById('stockChart');
    const traces = chart.data;
    const index = traces.findIndex(trace => trace.name === indicator);
    
    if (index > -1) {
        Plotly.deleteTraces('stockChart', index);
    }
}

// Timeframe Management
function handleTimeframeChange(e) {
    const timeframe = $(this).data('timeframe');
    $('.timeframe-btn').removeClass('active');
    $(this).addClass('active');
    
    showLoading();
    fetch(`/api/stock/${currentSymbol}/${timeframe}`)
        .then(response => response.json())
        .then(data => {
            updateChart(data.prices);
            updateIndicators(data.indicators);
        })
        .catch(error => showError('Failed to update timeframe'))
        .finally(() => hideLoading());
}

// Portfolio Management
function updatePortfolio() {
    showLoading();
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            renderPortfolio(data);
        })
        .catch(error => showError('Failed to update portfolio'))
        .finally(() => hideLoading());
}

function renderPortfolio(data) {
    const container = $('#portfolioContainer');
    container.empty();
    
    // Render summary
    const summary = `
        <div class="portfolio-summary">
            <h3>Portfolio Value: $${data.total_value.toFixed(2)}</h3>
            <p>Daily Change: ${data.daily_change}%</p>
        </div>
    `;
    container.append(summary);
    
    // Render positions
    data.positions.forEach(position => {
        const card = `
            <div class="card position-card">
                <div class="card-body">
                    <h5 class="card-title">${position.symbol}</h5>
                    <div class="row">
                        <div class="col">
                            <p>Shares: ${position.shares}</p>
                            <p>Cost Basis: $${position.cost_basis}</p>
                        </div>
                        <div class="col">
                            <p>Current Value: $${position.current_value}</p>
                            <p>P/L: ${position.profit_loss}%</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        container.append(card);
    });
}

// Watchlist Management
function updateWatchlist() {
    showLoading();
    fetch('/api/watchlist')
        .then(response => response.json())
        .then(data => {
            renderWatchlist(data);
        })
        .catch(error => showError('Failed to update watchlist'))
        .finally(() => hideLoading());
}

function renderWatchlist(data) {
    const container = $('#watchlistContainer');
    container.empty();
    
    data.items.forEach(item => {
        const row = `
            <div class="watchlist-item">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h5>${item.symbol}</h5>
                        <p class="mb-0">$${item.price}</p>
                    </div>
                    <div class="text-end">
                        <p class="mb-0 ${item.change >= 0 ? 'text-success' : 'text-danger'}">
                            ${item.change}%
                        </p>
                        <button class="btn btn-sm btn-danger remove-watchlist" data-symbol="${item.symbol}">
                            Remove
                        </button>
                    </div>
                </div>
            </div>
        `;
        container.append(row);
    });
    
    // Setup remove buttons
    $('.remove-watchlist').on('click', function() {
        const symbol = $(this).data('symbol');
        removeFromWatchlist(symbol);
    });
}

// User Preferences
function loadUserPreferences() {
    const darkMode = localStorage.getItem('darkMode') === 'true';
    if (darkMode) {
        $('body').addClass('dark-mode');
        $('#darkModeToggle').prop('checked', true);
    }
}

function handleDarkModeToggle() {
    const isDarkMode = $(this).prop('checked');
    $('body').toggleClass('dark-mode', isDarkMode);
    localStorage.setItem('darkMode', isDarkMode);
}

// Loading State Management
function showLoading() {
    $('#loadingOverlay').css('display', 'flex');
}

function hideLoading() {
    $('#loadingOverlay').css('display', 'none');
}

// Error Handling
function showError(message) {
    const alert = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    $('#errorContainer').html(alert);
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('Main.js loaded');
    
    const stockSelect = document.getElementById('stockSelect');
    const stockChart = document.getElementById('stockChart');
    const addStockButton = document.getElementById('addStockButton');
    const addStockModal = new bootstrap.Modal(document.getElementById('addStockModal'));
    const predictionPeriod = document.getElementById('predictionPeriod');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    function showLoading(show) {
        loadingOverlay.style.display = show ? 'flex' : 'none';
    }
    
    function updateChart(historyData) {
        const trace1 = {
            x: historyData.dates,
            y: historyData.prices,
            type: 'scatter',
            name: 'Stock Price',
            line: {
                color: '#17BECF'
            }
        };
        
        const trace2 = {
            x: historyData.dates,
            y: historyData.technical_indicators.sma20,
            type: 'scatter',
            name: 'SMA 20',
            line: {
                color: '#7F7F7F'
            }
        };
        
        const trace3 = {
            x: historyData.dates,
            y: historyData.technical_indicators.sma50,
            type: 'scatter',
            name: 'SMA 50',
            line: {
                color: '#FFA500'
            }
        };
        
        const trace4 = {
            x: historyData.dates,
            y: historyData.technical_indicators.upper_band,
            type: 'scatter',
            name: 'Upper Band',
            line: {
                color: '#FF9999',
                dash: 'dash'
            }
        };
        
        const trace5 = {
            x: historyData.dates,
            y: historyData.technical_indicators.lower_band,
            type: 'scatter',
            name: 'Lower Band',
            line: {
                color: '#FF9999',
                dash: 'dash'
            },
            fill: 'tonexty'
        };
        
        const data = [trace1, trace2, trace3, trace4, trace5];
        
        const layout = {
            title: 'Stock Price Analysis',
            xaxis: {
                title: 'Date',
                rangeslider: {visible: true}
            },
            yaxis: {
                title: 'Price ($)'
            }
        };
        
        Plotly.newPlot('stockChart', data, layout);
    }
    
    function updatePrices(predictionData) {
        document.getElementById('currentPrice').textContent = 
            `$${predictionData.current_price.toFixed(2)}`;
        document.getElementById('predictedPrice').textContent = 
            `$${predictionData.prediction.toFixed(2)}`;
            
        // Update prediction method and factors
        document.getElementById('predictionMethod').textContent = predictionData.method;
        
        const factorsList = document.getElementById('predictionFactors');
        factorsList.innerHTML = '';
        predictionData.factors.forEach(factor => {
            const li = document.createElement('li');
            li.textContent = factor;
            factorsList.appendChild(li);
        });
    }
    
    function updateTechnicalIndicators(historyData) {
        // Get the latest values
        const lastIndex = historyData.dates.length - 1;
        
        // Update RSI
        const rsiValue = historyData.technical_indicators.rsi[lastIndex];
        document.getElementById('rsiValue').textContent = 
            rsiValue ? rsiValue.toFixed(2) : '-';
        
        // Update SMA20
        const sma20Value = historyData.technical_indicators.sma20[lastIndex];
        document.getElementById('sma20Value').textContent = 
            sma20Value ? `$${sma20Value.toFixed(2)}` : '-';
        
        // Update SMA50
        const sma50Value = historyData.technical_indicators.sma50[lastIndex];
        document.getElementById('sma50Value').textContent = 
            sma50Value ? `$${sma50Value.toFixed(2)}` : '-';
    }
    
    function updateData() {
        const symbol = stockSelect.value;
        const period = predictionPeriod.value;
        
        showLoading(true);
        
        // Fetch historical data
        fetch(`/api/history/${symbol}`)
            .then(response => response.json())
            .then(historyData => {
                updateChart(historyData);
                updateTechnicalIndicators(historyData);
                
                // Fetch prediction with period
                return fetch(`/api/predict/${symbol}?period=${period}`);
            })
            .then(response => response.json())
            .then(predictionData => {
                updatePrices(predictionData);
            })
            .catch(error => {
                console.error('Error:', error);
                stockChart.innerHTML = '<div class="text-center text-danger">Error loading data</div>';
            })
            .finally(() => {
                showLoading(false);
            });
    }
    
    // Handle adding new stock
    addStockButton.addEventListener('click', function() {
        const symbolInput = document.getElementById('stockSymbol');
        const symbol = symbolInput.value.trim().toUpperCase();
        
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }
        
        showLoading(true);
        
        fetch('/api/add_stock', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add new option to select
                const option = new Option(symbol, symbol);
                stockSelect.add(option);
                
                // Select the new stock
                stockSelect.value = symbol;
                
                // Update the chart
                updateData();
                
                // Close modal
                addStockModal.hide();
                
                // Clear input
                symbolInput.value = '';
            } else {
                alert(data.error || 'Failed to add stock');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to add stock. Please try again.');
        })
        .finally(() => {
            showLoading(false);
        });
    });
    
    // Initial update
    updateData();
    
    // Update when stock selection changes
    stockSelect.addEventListener('change', updateData);
    
    // Update when prediction period changes
    predictionPeriod.addEventListener('change', updateData);
    
    // Update every 5 minutes
    setInterval(updateData, 300000);
});

console.log('Main.js loaded');

// Wait for DOM and all scripts to load
window.addEventListener('load', function() {
    try {
        console.log('Initializing application...');
        
        // Check if StockCharts is available
        if (typeof window.StockCharts !== 'function') {
            throw new Error('StockCharts class not loaded');
        }
        
        // Initialize charts
        window.charts = new window.StockCharts();
        console.log('StockCharts instance created');
        
        // Get initial symbol
        const stockSelect = document.getElementById('stockSelect');
        if (!stockSelect) {
            throw new Error('Stock select element not found');
        }
        
        window.currentSymbol = stockSelect.value;
        console.log('Initial symbol:', window.currentSymbol);
        
        // Initialize charts
        window.charts.initializePriceChart('stockChart');
        window.charts.initializeIndicatorChart('indicatorChart');
        
        // Set up event listeners
        stockSelect.addEventListener('change', function() {
            window.currentSymbol = this.value;
            console.log('Symbol changed to:', window.currentSymbol);
            updateData();
        });
        
        // Initial data load
        updateData();
        
        // Update data every 5 minutes
        setInterval(updateData, 300000);
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Error initializing application:', error);
        showError('Failed to initialize application. Please refresh the page.');
    }
});

function updateData() {
    try {
        console.log('Updating data for symbol:', window.currentSymbol);
        showLoading(true);
        
        // Fetch historical data
        fetch(`/api/history/${window.currentSymbol}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(historyData => {
                console.log('Received history data:', historyData);
                
                // Fetch prediction data
                return fetch(`/api/predict/${window.currentSymbol}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(predictionData => {
                        console.log('Received prediction data:', predictionData);
                        
                        // Update UI with both sets of data
                        updateUI(historyData, predictionData);
                    });
            })
            .catch(error => {
                console.error('Error fetching data:', error);
                showError('Failed to fetch data. Please try again later.');
            })
            .finally(() => {
                showLoading(false);
            });
    } catch (error) {
        console.error('Error in updateData:', error);
        showError('An error occurred while updating data.');
        showLoading(false);
    }
}

function updateUI(historyData, predictionData) {
    try {
        // Update charts
        window.charts.updatePriceChart({
            dates: historyData.dates,
            prices: historyData.prices,
            predictions: predictionData
        });
        
        window.charts.updateIndicatorChart({
            dates: historyData.dates,
            sma20: historyData.technical_indicators.sma20,
            sma50: historyData.technical_indicators.sma50,
            rsi: historyData.technical_indicators.rsi
        });
        
        // Update metrics
        const currentPriceElement = document.getElementById('currentPrice');
        const predictedPriceElement = document.getElementById('predictedPrice');
        const priceChangeElement = document.getElementById('priceChange');
        const lastUpdateElement = document.getElementById('lastUpdate');
        
        if (!currentPriceElement || !predictedPriceElement || !priceChangeElement || !lastUpdateElement) {
            throw new Error('Required UI elements not found');
        }
        
        currentPriceElement.textContent = `$${predictionData.current_price.toFixed(2)}`;
        predictedPriceElement.textContent = `$${predictionData.prediction.toFixed(2)}`;
        
        // Calculate and show price change
        const priceChange = predictionData.prediction - predictionData.current_price;
        const changePercent = (priceChange / predictionData.current_price * 100).toFixed(2);
        priceChangeElement.textContent = `${priceChange >= 0 ? '▲' : '▼'} ${Math.abs(changePercent)}%`;
        priceChangeElement.className = priceChange >= 0 ? 'price-up' : 'price-down';
        
        // Update last updated timestamp
        lastUpdateElement.textContent = new Date().toLocaleString();
        
        console.log('UI updated successfully');
    } catch (error) {
        console.error('Error updating UI:', error);
        showError('Failed to update display. Please refresh the page.');
    }
}

function showLoading(show) {
    const loadingElement = document.getElementById('loadingIndicator');
    if (loadingElement) {
        loadingElement.style.display = show ? 'block' : 'none';
    }
}

function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        setTimeout(() => {
            errorElement.style.display = 'none';
        }, 5000);
    } else {
        console.error('Error element not found:', message);
    }
}

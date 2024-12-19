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

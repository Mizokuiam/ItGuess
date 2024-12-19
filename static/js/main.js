document.addEventListener('DOMContentLoaded', function() {
    console.log('Main.js loaded');
    
    const stockSelect = document.getElementById('stockSelect');
    const stockChart = document.getElementById('stockChart');
    
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
    }
    
    function updateData() {
        const symbol = stockSelect.value;
        
        // Show loading state
        stockChart.innerHTML = '<div class="text-center">Loading...</div>';
        
        // Fetch historical data
        fetch(`/api/history/${symbol}`)
            .then(response => response.json())
            .then(historyData => {
                updateChart(historyData);
                
                // Fetch prediction
                return fetch(`/api/predict/${symbol}`);
            })
            .then(response => response.json())
            .then(predictionData => {
                updatePrices(predictionData);
            })
            .catch(error => {
                console.error('Error:', error);
                stockChart.innerHTML = '<div class="text-center text-danger">Error loading data</div>';
            });
    }
    
    // Initial update
    updateData();
    
    // Update when stock selection changes
    stockSelect.addEventListener('change', updateData);
    
    // Update every 5 minutes
    setInterval(updateData, 300000);
});

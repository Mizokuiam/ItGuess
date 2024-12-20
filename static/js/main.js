// Global Variables
let currentSymbol = '';
let stockChart = null;

document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('stockSearch');
    const searchButton = document.getElementById('searchButton');
    const searchResults = document.getElementById('searchResults');
    const stockData = document.getElementById('stockData');
    const periodSelect = document.getElementById('predictionPeriod');
    let currentSymbol = null;

    // Event listeners
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') performSearch();
    });
    periodSelect.addEventListener('change', function() {
        if (currentSymbol) updateStockData(currentSymbol);
    });

    function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        fetch(`/api/search_stock?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                searchResults.innerHTML = '';
                if (data.results && data.results.length > 0) {
                    data.results.forEach(stock => {
                        const item = document.createElement('a');
                        item.href = '#';
                        item.className = 'list-group-item list-group-item-action';
                        item.textContent = `${stock.symbol} - ${stock.name}`;
                        item.addEventListener('click', function(e) {
                            e.preventDefault();
                            searchResults.innerHTML = '';
                            searchInput.value = stock.symbol;
                            updateStockData(stock.symbol);
                        });
                        searchResults.appendChild(item);
                    });
                } else {
                    searchResults.innerHTML = '<div class="list-group-item">No results found</div>';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                searchResults.innerHTML = '<div class="list-group-item text-danger">Error searching for stocks</div>';
            });
    }

    function updateStockData(symbol) {
        currentSymbol = symbol;
        const period = periodSelect.value;
        stockData.classList.remove('d-none');
        
        document.getElementById('stockSymbol').textContent = symbol;

        fetch(`/api/stock/${symbol}?period=${period}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                // Update current price and change
                const priceElement = document.getElementById('currentPrice');
                const changeElement = document.getElementById('priceChange');
                const currentPrice = data.current.price;
                const priceChange = data.current.change;
                
                priceElement.textContent = `$${currentPrice.toFixed(2)}`;
                changeElement.textContent = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`;
                changeElement.className = priceChange >= 0 ? 'text-success' : 'text-danger';

                // Update technical indicators
                updateTechnicalIndicators(data.indicators);
                
                // Update predictions
                updatePredictions(data.predicted);
                
                // Update charts
                updateCharts(data);
            })
            .catch(error => {
                console.error('Error:', error);
                stockData.innerHTML = '<div class="alert alert-danger">Error fetching stock data</div>';
            });
    }

    function updateTechnicalIndicators(indicators) {
        // Update RSI
        document.getElementById('rsiValue').textContent = indicators.RSI;
        
        // Update MACD
        document.getElementById('macdValue').textContent = indicators.MACD;
        document.getElementById('signalValue').textContent = indicators.Signal;
        
        // Update EMAs
        document.getElementById('ema20Value').textContent = indicators.EMA20;
        document.getElementById('ema50Value').textContent = indicators.EMA50;
        
        // Update Bollinger Bands
        document.getElementById('bbUpperValue').textContent = indicators.BB_Upper;
        document.getElementById('bbMiddleValue').textContent = indicators.BB_Middle;
        document.getElementById('bbLowerValue').textContent = indicators.BB_Lower;
    }

    function updatePredictions(predictions) {
        // Update prediction values
        document.getElementById('ensemblePred').textContent = `$${predictions.ensemble}`;
        document.getElementById('lstmPred').textContent = `$${predictions.lstm}`;
        document.getElementById('rfPred').textContent = `$${predictions.rf}`;
        document.getElementById('lrPred').textContent = `$${predictions.lr}`;
        
        // Update R² scores
        if (predictions.metrics) {
            document.getElementById('lstmR2').textContent = predictions.metrics.lstm.r2.toFixed(3);
            document.getElementById('rfR2').textContent = predictions.metrics.rf.r2.toFixed(3);
            document.getElementById('lrR2').textContent = predictions.metrics.lr.r2.toFixed(3);
        }
        
        // Update confidence and date
        document.getElementById('predictionConfidence').textContent = predictions.confidence;
        document.getElementById('predictionDate').textContent = predictions.date;
    }

    function updateCharts(data) {
        // Price Chart
        const priceTrace = {
            x: data.prices.map(p => p.date),
            y: data.prices.map(p => p.price),
            type: 'scatter',
            name: 'Price'
        };
        
        Plotly.newPlot('priceChart', [priceTrace], {
            title: 'Stock Price History',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' }
        });

        // Technical Chart
        const technicalTraces = [
            {
                x: data.prices.map(p => p.date),
                y: data.prices.map(p => p.price),
                type: 'scatter',
                name: 'Price'
            },
            {
                x: data.prices.map(p => p.date),
                y: data.prices.map(p => p.ema20),
                type: 'scatter',
                name: 'EMA 20'
            },
            {
                x: data.prices.map(p => p.date),
                y: data.prices.map(p => p.ema50),
                type: 'scatter',
                name: 'EMA 50'
            },
            {
                x: data.prices.map(p => p.date),
                y: data.prices.map(p => p.bb_upper),
                type: 'scatter',
                name: 'BB Upper',
                line: { dash: 'dot' }
            },
            {
                x: data.prices.map(p => p.date),
                y: data.prices.map(p => p.bb_lower),
                type: 'scatter',
                name: 'BB Lower',
                line: { dash: 'dot' }
            }
        ];

        Plotly.newPlot('technicalChart', technicalTraces, {
            title: 'Technical Analysis',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' }
        });

        // Prediction Chart
        const predictionTraces = [
            {
                x: [data.prices[data.prices.length - 1].date, data.predicted.date],
                y: [data.current.price, data.predicted.ensemble],
                type: 'scatter',
                name: 'Ensemble',
                line: { dash: 'solid', width: 3 }
            },
            {
                x: [data.prices[data.prices.length - 1].date, data.predicted.date],
                y: [data.current.price, data.predicted.lstm],
                type: 'scatter',
                name: 'LSTM',
                line: { dash: 'dot' }
            },
            {
                x: [data.prices[data.prices.length - 1].date, data.predicted.date],
                y: [data.current.price, data.predicted.rf],
                type: 'scatter',
                name: 'Random Forest',
                line: { dash: 'dot' }
            },
            {
                x: [data.prices[data.prices.length - 1].date, data.predicted.date],
                y: [data.current.price, data.predicted.lr],
                type: 'scatter',
                name: 'Linear Regression',
                line: { dash: 'dot' }
            }
        ];

        Plotly.newPlot('predictionChart', predictionTraces, {
            title: 'Price Predictions',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' }
        });
    }
});

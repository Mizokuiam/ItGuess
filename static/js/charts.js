// Chart configuration and utilities
window.StockCharts = class {
    constructor() {
        console.log('Creating StockCharts instance');
        this.priceChart = null;
        this.indicatorChart = null;
        this.currentSymbol = null;
    }

    initializePriceChart(containerId) {
        try {
            console.log('Initializing price chart...');
            const container = document.getElementById(containerId);
            if (!container) {
                throw new Error(`Container ${containerId} not found`);
            }

            const layout = {
                title: 'Stock Price History',
                xaxis: {
                    title: 'Date',
                    rangeslider: { visible: true }
                },
                yaxis: {
                    title: 'Price ($)'
                },
                showlegend: true,
                height: 400
            };

            this.priceChart = Plotly.newPlot(containerId, [], layout);
            console.log('Price chart initialized');
        } catch (error) {
            console.error('Error initializing price chart:', error);
            throw error;
        }
    }

    initializeIndicatorChart(containerId) {
        try {
            console.log('Initializing indicator chart...');
            const container = document.getElementById(containerId);
            if (!container) {
                throw new Error(`Container ${containerId} not found`);
            }

            const layout = {
                title: 'Technical Indicators',
                xaxis: { title: 'Date' },
                yaxis: { 
                    title: 'Moving Averages',
                    domain: [0.6, 1]
                },
                yaxis2: {
                    title: 'RSI',
                    domain: [0, 0.4],
                    range: [0, 100]
                },
                showlegend: true,
                height: 400,
                grid: { rows: 2, columns: 1, pattern: 'independent' }
            };

            this.indicatorChart = Plotly.newPlot(containerId, [], layout);
            console.log('Indicator chart initialized');
        } catch (error) {
            console.error('Error initializing indicator chart:', error);
            throw error;
        }
    }

    updatePriceChart(data) {
        try {
            console.log('Updating price chart with data:', data);
            if (!data || !data.dates || !data.prices) {
                throw new Error('Invalid data format for price chart');
            }

            const traces = [{
                x: data.dates,
                y: data.prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Actual Price',
                line: { color: '#1f77b4' }
            }];

            if (data.predictions) {
                traces.push({
                    x: [data.dates[data.dates.length - 1]],
                    y: [data.predictions.prediction],
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Predicted Price',
                    marker: {
                        size: 10,
                        color: '#ff7f0e'
                    }
                });
            }

            Plotly.react('stockChart', traces);
            console.log('Price chart updated successfully');
        } catch (error) {
            console.error('Error updating price chart:', error);
            throw error;
        }
    }

    updateIndicatorChart(data) {
        try {
            console.log('Updating indicator chart with data:', data);
            if (!data || !data.dates || !data.sma20 || !data.sma50 || !data.rsi) {
                throw new Error('Invalid data format for indicator chart');
            }

            const traces = [
                {
                    x: data.dates,
                    y: data.sma20,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA (20)',
                    line: { color: '#2ca02c' }
                },
                {
                    x: data.dates,
                    y: data.sma50,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA (50)',
                    line: { color: '#d62728' }
                },
                {
                    x: data.dates,
                    y: data.rsi,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'RSI',
                    yaxis: 'y2',
                    line: { color: '#9467bd' }
                }
            ];

            Plotly.react('indicatorChart', traces);
            console.log('Indicator chart updated successfully');
        } catch (error) {
            console.error('Error updating indicator chart:', error);
            throw error;
        }
    }

    updateMetrics(metrics) {
        document.getElementById('currentPrice').textContent = `$${metrics.current_price.toFixed(2)}`;
        document.getElementById('predictedPrice').textContent = `$${metrics.predicted_price.toFixed(2)}`;
        
        const priceChange = metrics.predicted_price - metrics.current_price;
        const changePercent = (priceChange / metrics.current_price * 100).toFixed(2);
        const direction = priceChange >= 0 ? 'up' : 'down';
        
        document.getElementById('priceChange').textContent = 
            `${direction === 'up' ? '▲' : '▼'} ${Math.abs(changePercent)}%`;
        document.getElementById('priceChange').className = `price-${direction}`;
    }

    updateTechnicalIndicators(data) {
        const indicators = document.getElementById('technicalIndicators');
        indicators.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6>RSI</h6>
                            <p>${data.rsi.toFixed(2)}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6>SMA (20)</h6>
                            <p>$${data.sma20.toFixed(2)}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6>SMA (50)</h6>
                            <p>$${data.sma50.toFixed(2)}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
};

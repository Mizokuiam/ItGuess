// Global Variables
let currentSymbol = '';
let chartInstance = null;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Main.js loaded');
    
    // Initialize Select2
    $(document).ready(function() {
        $('#stockSelect').select2({
            theme: 'bootstrap4',
            placeholder: 'Select a stock...',
            allowClear: true
        });

        // Initialize Chart
        const ctx = document.getElementById('priceChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Stock Price',
                    data: [],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#2196F3',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Price ($)'
                        }
                    }
                }
            }
        });

        // Event Listeners
        $('#stockSelect').on('change', function() {
            const selectedStock = $(this).val();
            if (selectedStock) {
                updateData(selectedStock);
            }
        });

        $('#darkModeToggle').on('change', function() {
            $('body').toggleClass('dark-mode');
            updateChartTheme();
        });

        // Form submission
        $('#addStockForm').on('submit', function(e) {
            e.preventDefault();
            const symbol = $('#stockSymbol').val().toUpperCase();
            addNewStock(symbol);
        });
    });

    // Update chart theme based on dark mode
    function updateChartTheme() {
        const isDarkMode = $('body').hasClass('dark-mode');
        const textColor = isDarkMode ? '#ffffff' : '#666666';
        
        chartInstance.options.scales.x.ticks.color = textColor;
        chartInstance.options.scales.y.ticks.color = textColor;
        chartInstance.options.scales.x.title.color = textColor;
        chartInstance.options.scales.y.title.color = textColor;
        chartInstance.options.plugins.legend.labels.color = textColor;
        chartInstance.update();
    }

    // Fetch and update stock data
    async function updateData(symbol) {
        showLoading();
        try {
            const period = document.getElementById('predictionPeriod').value;
            const response = await fetch(`/api/stock/${symbol}?period=${period}`);
            const data = await response.json();
            
            if (data.success) {
                updateChart(data.prices, data.dates);
                updatePrices({
                    current: data.current_price,
                    previous: data.previous_close,
                    predicted: data.prediction.price,
                    confidence: data.prediction.confidence
                });
                updateTechnicalIndicators(data.indicators);
            } else {
                showError('Failed to fetch stock data');
            }
        } catch (error) {
            showError('Error fetching stock data');
            console.error('Error:', error);
        } finally {
            hideLoading();
        }
    }

    // Update chart with new data
    function updateChart(prices, dates) {
        chartInstance.data.labels = dates;
        chartInstance.data.datasets[0].data = prices;
        chartInstance.update();
    }

    // Get and display prediction
    async function getPrediction(symbol) {
        showLoading();
        try {
            const response = await fetch(`/api/predict/${symbol}`);
            const data = await response.json();
            
            if (data.success) {
                updatePredictionDisplay(data);
            } else {
                showError('Failed to get prediction');
            }
        } catch (error) {
            showError('Error getting prediction');
            console.error('Error:', error);
        } finally {
            hideLoading();
        }
    }

    // Update prediction display
    function updatePredictionDisplay(data) {
        const currentPrice = parseFloat(data.current_price).toFixed(2);
        const predictedPrice = parseFloat(data.predicted_price).toFixed(2);
        const change = ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
        
        $('#currentPrice').text(`$${currentPrice}`);
        $('#predictedPrice').text(`$${predictedPrice}`);
        
        const changeElement = $('#priceChange');
        changeElement.text(`${change}%`);
        changeElement.removeClass('price-up price-down');
        changeElement.addClass(change >= 0 ? 'price-up' : 'price-down');
    }

    // Add new stock to the list
    async function addNewStock(symbol) {
        showLoading();
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
                $('#stockSelect').append(new Option(data.name, symbol, true, true)).trigger('change');
                $('#addStockModal').modal('hide');
            } else {
                showError(data.message || 'Failed to add stock');
            }
        } catch (error) {
            showError('Error adding stock');
            console.error('Error:', error);
        } finally {
            hideLoading();
        }
    }

    // Loading state management
    function showLoading() {
        $('#loadingOverlay').fadeIn();
    }

    function hideLoading() {
        $('#loadingOverlay').fadeOut();
    }

    // Error handling
    function showError(message) {
        // You can implement a toast or alert system here
        alert(message);
    }

    // Update technical indicators
    function updateTechnicalIndicators(data) {
        const indicators = $('#technicalIndicators');
        indicators.empty();
        
        const indicatorData = [
            { name: 'RSI', value: data.rsi, type: data.rsi > 70 ? 'danger' : data.rsi < 30 ? 'success' : 'info' },
            { name: 'MACD', value: data.macd, type: data.macd > 0 ? 'success' : 'danger' },
            { name: 'Volume', value: data.volume.toLocaleString(), type: 'info' }
        ];
        
        indicatorData.forEach(indicator => {
            const card = `
                <div class="col-md-4">
                    <div class="indicator-card card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${indicator.name}</h5>
                            <p class="indicator-value text-${indicator.type}">${indicator.value}</p>
                        </div>
                    </div>
                </div>
            `;
            indicators.append(card);
        });
    }

    // Update prices
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
});

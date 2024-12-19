// Global Variables
let currentSymbol = '';
let stockChart = null;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Main.js loaded');
    
    // Initialize Select2
    $(document).ready(function() {
        // Initialize Select2 for stock selection
        $('#stockSelect').select2({
            theme: 'bootstrap4',
            placeholder: 'Select a stock...',
            allowClear: true
        });

        // Initialize Chart
        const ctx = document.getElementById('stockChart').getContext('2d');
        stockChart = new Chart(ctx, {
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
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#666666'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#2196F3',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date',
                            color: '#666666'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Price ($)',
                            color: '#666666'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                animation: {
                    duration: 1000
                }
            }
        });

        // Event Listeners
        $('#stockSelect').on('change', function() {
            const selectedStock = $(this).val();
            if (selectedStock) {
                updateStockData(selectedStock);
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

        // Initialize tooltips
        $('[data-bs-toggle="tooltip"]').tooltip();
    });

    // Update chart theme based on dark mode
    function updateChartTheme() {
        const isDarkMode = $('body').hasClass('dark-mode');
        const textColor = isDarkMode ? '#ffffff' : '#666666';
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        
        stockChart.options.scales.x.ticks.color = textColor;
        stockChart.options.scales.y.ticks.color = textColor;
        stockChart.options.scales.x.title.color = textColor;
        stockChart.options.scales.y.title.color = textColor;
        stockChart.options.scales.x.grid.color = gridColor;
        stockChart.options.scales.y.grid.color = gridColor;
        stockChart.options.plugins.legend.labels.color = textColor;
        stockChart.update();
    }

    // Fetch and update stock data
    async function updateStockData(symbol) {
        showLoading();
        try {
            const response = await fetch(`/api/stock/${symbol}`);
            const data = await response.json();
            
            if (data.success) {
                updateChart(data.prices, data.dates);
                updateCurrentPrice(data.current_price, data.previous_close);
                getPrediction(symbol);
                updateTechnicalIndicators(data.indicators);
            } else {
                showError(data.error || 'Failed to fetch stock data');
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
        stockChart.data.labels = dates;
        stockChart.data.datasets[0].data = prices;
        stockChart.update();
    }

    // Update current price display
    function updateCurrentPrice(currentPrice, previousClose) {
        const change = ((currentPrice - previousClose) / previousClose * 100).toFixed(2);
        const direction = change >= 0 ? 'up' : 'down';
        
        $('#currentPrice').text(`$${currentPrice}`);
        $('#priceChange').html(`
            <i class="fas fa-arrow-${direction}"></i>
            ${Math.abs(change)}%
        `).removeClass('text-success text-danger')
          .addClass(direction === 'up' ? 'text-success' : 'text-danger');
    }

    // Get and display prediction
    async function getPrediction(symbol) {
        try {
            const response = await fetch(`/api/predict/${symbol}`);
            const data = await response.json();
            
            if (data.success) {
                updatePredictionDisplay(data);
            } else {
                showError(data.error || 'Failed to get prediction');
            }
        } catch (error) {
            showError('Error getting prediction');
            console.error('Error:', error);
        }
    }

    // Update prediction display
    function updatePredictionDisplay(data) {
        const currentPrice = parseFloat(data.current_price);
        const predictedPrice = parseFloat(data.predicted_price);
        const change = ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
        const direction = change >= 0 ? 'up' : 'down';
        
        $('#predictedPrice').text(`$${predictedPrice}`);
        $('#predictionChange').html(`
            <i class="fas fa-arrow-${direction}"></i>
            ${Math.abs(change)}%
        `).removeClass('text-success text-danger')
          .addClass(direction === 'up' ? 'text-success' : 'text-danger');
        
        $('#predictionDate').text(`Prediction for: ${data.prediction_date}`);
        $('#predictionConfidence').text(`Confidence: ${(data.confidence * 100).toFixed(1)}%`);
    }

    // Update technical indicators
    function updateTechnicalIndicators(data) {
        const indicators = $('#technicalIndicators');
        indicators.empty();
        
        const indicatorData = [
            {
                name: 'RSI',
                value: data.rsi,
                icon: 'fa-signal',
                description: 'Relative Strength Index',
                type: data.rsi > 70 ? 'danger' : data.rsi < 30 ? 'success' : 'info'
            },
            {
                name: 'MACD',
                value: data.macd,
                icon: 'fa-chart-line',
                description: 'Moving Average Convergence Divergence',
                type: data.macd > 0 ? 'success' : 'danger'
            },
            {
                name: 'Signal',
                value: data.signal,
                icon: 'fa-wave-square',
                description: 'MACD Signal Line',
                type: 'info'
            }
        ];
        
        indicatorData.forEach(indicator => {
            if (indicator.value !== null) {
                const card = `
                    <div class="col-md-4">
                        <div class="card indicator-card">
                            <div class="card-body text-center">
                                <i class="fas ${indicator.icon} mb-2 text-${indicator.type}"></i>
                                <h5 class="card-title">${indicator.name}</h5>
                                <p class="indicator-value text-${indicator.type}">${indicator.value}</p>
                                <small class="text-muted">${indicator.description}</small>
                            </div>
                        </div>
                    </div>
                `;
                indicators.append(card);
            }
        });
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
        $('#loadingOverlay').fadeIn(200);
    }

    function hideLoading() {
        $('#loadingOverlay').fadeOut(200);
    }

    // Error handling with toast notifications
    function showError(message) {
        const toast = `
            <div class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="3000">
                <div class="toast-header bg-danger text-white">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    <strong class="me-auto">Error</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        const toastContainer = $('#toastContainer');
        if (!toastContainer.length) {
            $('body').append('<div id="toastContainer" class="toast-container position-fixed top-0 end-0 p-3"></div>');
        }
        
        const toastElement = $(toast);
        $('#toastContainer').append(toastElement);
        const bsToast = new bootstrap.Toast(toastElement[0]);
        bsToast.show();
    }
});

// Global Variables
let currentSymbol = '';
let stockChart = null;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Main.js loaded');
    
    $(document).ready(function() {
        // Initialize Select2 for stock selection
        $('#stockSelect').select2({
            placeholder: 'Choose a stock...',
            allowClear: true
        });

        // Initialize dark mode based on user preference
        const darkModeToggle = $('#darkModeToggle');
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        darkModeToggle.prop('checked', isDarkMode);
        if (isDarkMode) {
            $('body').addClass('dark-mode');
        }

        // Dark mode toggle handler
        darkModeToggle.on('change', function() {
            const isChecked = $(this).prop('checked');
            localStorage.setItem('darkMode', isChecked);
            $('body').toggleClass('dark-mode', isChecked);
        });

        // Stock chart initialization
        let stockChart = null;
        function initializeChart() {
            const ctx = document.getElementById('stockChart').getContext('2d');
            stockChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Stock Price',
                        data: [],
                        borderColor: '#2196F3',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        initializeChart();

        // Add stock handler
        $('#addStockButton').on('click', function() {
            const symbol = $('#newStockSymbol').val().trim().toUpperCase();
            if (symbol) {
                // Show loading overlay
                $('#loadingOverlay').css('display', 'flex');

                // Make API call to add stock
                $.ajax({
                    url: '/api/add_stock',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ symbol: symbol }),
                    success: function(response) {
                        if (response.success) {
                            // Add new option to select
                            const newOption = new Option(symbol, symbol, true, true);
                            $('#stockSelect').append(newOption).trigger('change');
                            
                            // Close modal and clear input
                            $('#addStockModal').modal('hide');
                            $('#newStockSymbol').val('');
                            
                            // Load the new stock data
                            loadStockData(symbol);
                        } else {
                            alert('Error adding stock: ' + response.message);
                        }
                    },
                    error: function() {
                        alert('Error adding stock. Please try again.');
                    },
                    complete: function() {
                        $('#loadingOverlay').hide();
                    }
                });
            }
        });

        // Stock selection handler
        $('#stockSelect').on('change', function() {
            const symbol = $(this).val();
            if (symbol) {
                loadStockData(symbol);
            }
        });

        // Prediction period handler
        $('#predictionPeriod').on('change', function() {
            const symbol = $('#stockSelect').val();
            if (symbol) {
                loadStockData(symbol);
            }
        });

        // Load stock data function
        function loadStockData(symbol) {
            $('#loadingOverlay').css('display', 'flex');
            const period = $('#predictionPeriod').val();

            $.ajax({
                url: '/api/stock_data',
                method: 'GET',
                data: { 
                    symbol: symbol,
                    period: period
                },
                success: function(data) {
                    updateChart(data.prices);
                    updatePriceDisplays(data.current, data.predicted);
                    updateTechnicalIndicators(data.indicators);
                },
                error: function() {
                    alert('Error loading stock data. Please try again.');
                },
                complete: function() {
                    $('#loadingOverlay').hide();
                }
            });
        }

        // Update chart function
        function updateChart(prices) {
            stockChart.data.labels = prices.map(p => p.date);
            stockChart.data.datasets[0].data = prices.map(p => p.price);
            stockChart.update();
        }

        // Update price displays function
        function updatePriceDisplays(current, predicted) {
            $('#currentPrice').text('$' + current.price.toFixed(2));
            $('#predictedPrice').text('$' + predicted.price.toFixed(2));
            
            const priceChange = current.change;
            const changeText = (priceChange >= 0 ? '+' : '') + priceChange.toFixed(2) + '%';
            $('#priceChange')
                .text(changeText)
                .removeClass('price-up price-down')
                .addClass(priceChange >= 0 ? 'price-up' : 'price-down');
            
            $('#confidenceLevel').text('Confidence: ' + predicted.confidence + '%');
            $('#volume').text(formatNumber(current.volume));
        }

        // Update technical indicators function
        function updateTechnicalIndicators(indicators) {
            const container = $('#technicalIndicators');
            container.empty();

            Object.entries(indicators).forEach(([name, value]) => {
                const card = $('<div>').addClass('col-md-4 mb-3');
                card.html(`
                    <div class="card">
                        <div class="card-body text-center">
                            <h6 class="card-title">${name}</h6>
                            <p class="mb-0">${value}</p>
                        </div>
                    </div>
                `);
                container.append(card);
            });
        }

        // Utility function to format large numbers
        function formatNumber(num) {
            return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        }
    });
});

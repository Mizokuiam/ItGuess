import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import csv
from datetime import datetime

class ExportService:
    @staticmethod
    def export_to_csv(data, filename):
        """Export data to CSV format"""
        try:
            df = pd.DataFrame(data)
            output = BytesIO()
            df.to_csv(output, index=False, quoting=csv.QUOTE_NONNUMERIC)
            output.seek(0)
            return True, output
        except Exception as e:
            return False, str(e)
            
    @staticmethod
    def generate_portfolio_report(portfolio_data, user_data):
        """Generate PDF report for portfolio"""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            elements.append(Paragraph("Portfolio Report", title_style))
            elements.append(Spacer(1, 12))
            
            # Portfolio Summary
            elements.append(Paragraph("Portfolio Summary", styles['Heading2']))
            summary_data = [
                ["Total Value", f"${portfolio_data['total_value']:,.2f}"],
                ["Number of Positions", str(len(portfolio_data['positions']))],
                ["Report Date", datetime.now().strftime("%Y-%m-%d %H:%M")]
            ]
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Positions Detail
            elements.append(Paragraph("Positions Detail", styles['Heading2']))
            positions_data = [["Symbol", "Shares", "Entry Price", "Current Price", "Value", "P/L %"]]
            for position in portfolio_data['positions']:
                positions_data.append([
                    position['symbol'],
                    f"{position['shares']:,.2f}",
                    f"${position['entry_price']:,.2f}",
                    f"${position['current_price']:,.2f}",
                    f"${position['position_value']:,.2f}",
                    f"{position['profit_loss_percent']:,.2f}%"
                ])
            
            positions_table = Table(positions_data)
            positions_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(positions_table)
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            return True, buffer
        except Exception as e:
            return False, str(e)
            
    @staticmethod
    def generate_analysis_report(stock_data, analysis_data):
        """Generate PDF report for stock analysis"""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            elements.append(Paragraph(f"Stock Analysis Report - {stock_data['symbol']}", title_style))
            elements.append(Spacer(1, 12))
            
            # Price Information
            elements.append(Paragraph("Price Information", styles['Heading2']))
            price_data = [
                ["Current Price", f"${stock_data['current_price']:,.2f}"],
                ["52 Week High", f"${stock_data['year_high']:,.2f}"],
                ["52 Week Low", f"${stock_data['year_low']:,.2f}"],
                ["Volume", f"{stock_data['volume']:,}"]
            ]
            price_table = Table(price_data)
            price_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(price_table)
            elements.append(Spacer(1, 20))
            
            # Technical Indicators
            elements.append(Paragraph("Technical Indicators", styles['Heading2']))
            indicators_data = [
                ["Indicator", "Value", "Signal"],
                ["RSI", f"{analysis_data['rsi']:,.2f}", analysis_data['rsi_signal']],
                ["MACD", f"{analysis_data['macd']:,.2f}", analysis_data['macd_signal']],
                ["SMA20", f"${analysis_data['sma20']:,.2f}", analysis_data['sma_signal']]
            ]
            indicators_table = Table(indicators_data)
            indicators_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(indicators_table)
            elements.append(Spacer(1, 20))
            
            # Prediction
            elements.append(Paragraph("Price Prediction", styles['Heading2']))
            prediction_data = [
                ["Time Frame", "Predicted Price", "Confidence"],
                ["Short Term", f"${analysis_data['short_term_prediction']:,.2f}", f"{analysis_data['short_term_confidence']:,.1f}%"],
                ["Medium Term", f"${analysis_data['medium_term_prediction']:,.2f}", f"{analysis_data['medium_term_confidence']:,.1f}%"],
                ["Long Term", f"${analysis_data['long_term_prediction']:,.2f}", f"{analysis_data['long_term_confidence']:,.1f}%"]
            ]
            prediction_table = Table(prediction_data)
            prediction_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(prediction_table)
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            return True, buffer
        except Exception as e:
            return False, str(e)

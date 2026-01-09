
"""
Data handling utilities for Portfolio Risk Analysis
The Mountain Path - World of Finance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache

class DataHandler:
    """Handle data fetching and caching"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def fetch_stock_data(tickers, start_date, end_date):
        """
        Fetch stock data with robust column handling for yfinance 0.2.50+
        """
        try:
            # Convert tickers to list if passed as tuple (from lru_cache)
            ticker_list = list(tickers)
            
            # Download data with auto_adjust=False to attempt getting separate Adj Close
            raw_data = yf.download(
                ticker_list,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            if raw_data.empty:
                return None

            # 1. Handle MultiIndex (Multiple Tickers)
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Try to get Adj Close level
                if 'Adj Close' in raw_data.columns.levels[0]:
                    data = raw_data['Adj Close']
                else:
                    data = raw_data['Close']
            
            # 2. Handle Single Index (Single Ticker)
            else:
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close']
                else:
                    data = raw_data['Close']
            
            # 3. Force consistency: Ensure result is always a DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = ticker_list
            
            # Drop any completely empty columns or rows
            data = data.ffill().dropna()
            
            return data
            
        except Exception as e:
            print(f"Error fetching data in DataHandler: {e}")
            return None
    
    @staticmethod
    def validate_weights(weights_dict):
        """Check if portfolio weights sum to 100%"""
        total = sum(weights_dict.values())
        return abs(total - 100.0) < 0.1

    @staticmethod
    def prepare_export_data(portfolio_data, metrics, composition_df):
        """
        Format data structures for Excel export
        """
        metrics_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Percent formatting for specific keys
                if any(x in key for x in ['Return', 'Alpha', 'Drawdown', 'VaR', 'ES']):
                    formatted_value = f"{value*100:.2f}%"
                # Float formatting for ratios
                elif any(x in key for x in ['Ratio', 'Beta', 'Skew', 'Kurt']):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.4f}"
                
                metrics_rows.append({'Metric': key, 'Value': formatted_value})
        
        return {
            'Portfolio Composition': composition_df,
            'Risk Metrics': pd.DataFrame(metrics_rows),
            'Daily Returns': portfolio_data['portfolio_returns'].to_frame(name='Portfolio Daily Return')
        }

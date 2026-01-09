
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
        Fetch stock data with robust column handling
        """
        try:
            # Download data with auto_adjust=False to attempt getting Adj Close
            raw_data = yf.download(
                list(tickers),
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            if raw_data.empty:
                return None

            # Handle MultiIndex columns (standard for multiple tickers)
            if isinstance(raw_data.columns, pd.MultiIndex):
                if 'Adj Close' in raw_data.columns.levels[0]:
                    data = raw_data['Adj Close']
                else:
                    data = raw_data['Close']
            else:
                # Handle Single Index (standard for one ticker)
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close']
                else:
                    data = raw_data['Close']
            
            # Ensure output is a DataFrame even for 1 ticker
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = list(tickers)
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    @staticmethod
    def validate_weights(weights_dict):
        total = sum(weights_dict.values())
        return abs(total - 100.0) < 0.1

    @staticmethod
    def prepare_export_data(portfolio_data, metrics, composition_df):
        metrics_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'Return' in key or 'Alpha' in key or 'Drawdown' in key:
                    formatted_value = f"{value*100:.2f}%"
                elif 'Ratio' in key or 'Beta' in key:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.4f}"
                
                metrics_rows.append({'Metric': key, 'Value': formatted_value})
        
        return {
            'Portfolio Composition': composition_df,
            'Risk Metrics': pd.DataFrame(metrics_rows),
            'Daily Returns': portfolio_data['portfolio_returns']
        }

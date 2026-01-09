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
        Fetch stock data with caching
        
        Parameters:
        -----------
        tickers : tuple
            Stock symbols to fetch
        start_date : str
            Start date for data
        end_date : str
            End date for data
            
        Returns:
        --------
        pd.DataFrame
            Adjusted close prices
        """
        try:
            data = yf.download(
                list(tickers),
                start=start_date,
                end=end_date,
                progress=False
            )['Adj Close']
            
            if len(tickers) == 1:
                data = pd.DataFrame(data)
                data.columns = list(tickers)
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    @staticmethod
    def validate_weights(weights_dict):
        """
        Validate portfolio weights
        
        Parameters:
        -----------
        weights_dict : dict
            Dictionary of stock weights
            
        Returns:
        --------
        bool
            True if valid, False otherwise
        """
        total = sum(weights_dict.values())
        return abs(total - 100) <= 0.01
    
    @staticmethod
    def prepare_export_data(metrics, portfolio_data, composition_df):
        """
        Prepare data for Excel export
        
        Parameters:
        -----------
        metrics : dict
            Calculated risk metrics
        portfolio_data : dict
            Portfolio data
        composition_df : pd.DataFrame
            Portfolio composition
            
        Returns:
        --------
        dict
            Dictionary of DataFrames for export
        """
        # Create metrics DataFrame
        metrics_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'Return' in key or 'Alpha' in key:
                    formatted_value = f"{value*100:.2f}%"
                elif 'Ratio' in key or 'Beta' in key or 'Skew' in key or 'Kurt' in key:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value*100:.2f}%"
                
                metrics_rows.append({
                    'Metric': key,
                    'Value': formatted_value
                })
        
        metrics_df = pd.DataFrame(metrics_rows)
        
        return {
            'Portfolio Composition': composition_df,
            'Risk Metrics': metrics_df,
            'Summary Statistics': pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    portfolio_data['portfolio_returns'].mean(),
                    portfolio_data['portfolio_returns'].std(),
                    portfolio_data['portfolio_returns'].min(),
                    portfolio_data['portfolio_returns'].max(),
                    metrics.get('Skewness', 0),
                    metrics.get('Kurtosis', 0)
                ]
            })
        }

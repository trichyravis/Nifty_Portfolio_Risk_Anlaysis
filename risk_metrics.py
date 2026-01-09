
"""
Risk Metrics Calculation Functions
The Mountain Path - World of Finance
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_portfolio_returns(prices_df, weights):
    """Calculate portfolio returns"""
    returns = prices_df.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    return returns, portfolio_returns

def calculate_var(returns, confidence_level=0.95):
    """Calculate Value at Risk"""
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_expected_shortfall(returns, confidence_level=0.95):
    """Calculate Expected Shortfall (Conditional VaR)"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_maximum_drawdown(returns):
    """Calculate Maximum Drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min(), drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.065):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.065):
    """Calculate Sortino Ratio"""
    excess_returns = returns - risk_free_rate/252
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 0
    
    return np.sqrt(252) * excess_returns.mean() / downside_std

def calculate_calmar_ratio(returns):
    """Calculate Calmar Ratio"""
    annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
    max_dd, _ = calculate_maximum_drawdown(returns)
    
    if max_dd == 0:
        return 0
    
    return annual_return / abs(max_dd)

def calculate_information_ratio(returns, benchmark_returns):
    """Calculate Information Ratio"""
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    
    if tracking_error == 0:
        return 0
    
    return np.sqrt(252) * active_returns.mean() / tracking_error

def calculate_portfolio_metrics(prices_df, weights, benchmark_returns=None, risk_free_rate=0.065):
    """Calculate all portfolio metrics"""
    returns, portfolio_returns = calculate_portfolio_returns(prices_df, weights)
    
    metrics = {
        'Annual Return': (1 + portfolio_returns).prod() ** (252/len(portfolio_returns)) - 1,
        'Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(portfolio_returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(portfolio_returns, risk_free_rate),
        'Calmar Ratio': calculate_calmar_ratio(portfolio_returns),
        'VaR 95%': calculate_var(portfolio_returns, 0.95),
        'VaR 99%': calculate_var(portfolio_returns, 0.99),
        'ES 95%': calculate_expected_shortfall(portfolio_returns, 0.95),
        'ES 99%': calculate_expected_shortfall(portfolio_returns, 0.99),
        'Max Drawdown': calculate_maximum_drawdown(portfolio_returns)[0],
        'Skewness': stats.skew(portfolio_returns),
        'Kurtosis': stats.kurtosis(portfolio_returns)
    }
    
    if benchmark_returns is not None:
        metrics['Information Ratio'] = calculate_information_ratio(portfolio_returns, benchmark_returns)
        metrics['Beta'] = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
        metrics['Alpha'] = metrics['Annual Return'] - (risk_free_rate + metrics['Beta'] * (benchmark_returns.mean() * 252 - risk_free_rate))
    
    return metrics, portfolio_returns

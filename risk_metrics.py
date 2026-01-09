
import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_portfolio_metrics(stock_data, weights_dict, benchmark_returns=None, risk_free_rate=0.05):
    """
    Calculate comprehensive risk and performance metrics
    """
    # Align weights with columns
    weights = np.array([weights_dict[ticker] / 100 for ticker in stock_data.columns])
    
    # Calculate daily returns
    returns = stock_data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    
    # Annualization factor
    ann_factor = 252
    
    # Performance Metrics
    total_return = (1 + portfolio_returns).prod() - 1
    ann_return = portfolio_returns.mean() * ann_factor
    volatility = portfolio_returns.std() * np.sqrt(ann_factor)
    
    # Risk-Adjusted Ratios
    sharpe = (ann_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    # Downside Risk (Sortino)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(ann_factor)
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
    
    # Maximum Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Value at Risk (Historical)
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    
    # Expected Shortfall (Conditional VaR)
    es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    metrics = {
        'Annual Return': ann_return,
        'Cumulative Return': total_return,
        'Annual Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': ann_return / abs(max_drawdown) if max_drawdown != 0 else 0,
        'VaR 95%': var_95,
        'VaR 99%': var_99,
        'ES 95%': es_95,
        'ES 99%': es_99,
        'Skewness': portfolio_returns.skew(),
        'Kurtosis': portfolio_returns.kurtosis()
    }
    
    return metrics, portfolio_returns

def calculate_maximum_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min(), drawdown

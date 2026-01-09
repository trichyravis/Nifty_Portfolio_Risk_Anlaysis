
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

def calculate_portfolio_metrics(stock_data, weights, benchmark_returns, risk_free_rate):
    # Calculate daily returns
    returns = stock_data.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Annualized metrics
    ann_return = portfolio_returns.mean() * 252
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Risk-adjusted ratios
    excess_return = ann_return - risk_free_rate
    sharpe = excess_return / volatility if volatility != 0 else 0
    
    # Downside risk for Sortino
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = excess_return / downside_vol if downside_vol != 0 else 0
    
    # VaR and ES (Historical)
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    # Max Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'Annual Return': ann_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': ann_return / abs(max_drawdown) if max_drawdown != 0 else 0,
        'VaR 95%': var_95,
        'VaR 99%': var_99,
        'ES 95%': es_95,
        'ES 99%': es_99,
        'Skewness': skew(portfolio_returns),
        'Kurtosis': kurtosis(portfolio_returns)
    }
    
    return metrics, portfolio_returns

def calculate_maximum_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min(), drawdown

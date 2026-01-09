
"""
Risk Metrics and Performance Analytics Engine
The Mountain Path - World of Finance
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_portfolio_metrics(stock_data, weights_input, benchmark_returns=None, risk_free_rate=0.05):
    """
    Calculate comprehensive risk and performance metrics for the portfolio.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Adjusted close prices of stocks
    weights_input : dict or np.array
        Weights for each stock (dict with tickers as keys or array in order of columns)
    benchmark_returns : pd.Series, optional
        Returns of the benchmark index
    risk_free_rate : float
        Annual risk-free rate (decimal, e.g., 0.05 for 5%)
        
    Returns:
    --------
    dict, pd.Series
        Dictionary of calculated metrics and the portfolio returns series
    """
    
    # 1. Align weights with stock_data columns
    if isinstance(weights_input, dict):
        # Extract weights based on the column order of stock_data
        w = np.array([weights_input.get(ticker, 0) for ticker in stock_data.columns])
    else:
        # Assume it's already an array/list aligned with columns
        w = np.array(weights_input)
    
    # Ensure weights are normalized (sum to 1.0)
    if w.sum() != 1.0:
        w = w / w.sum()

    # 2. Calculate daily returns (Simple Returns)
    returns = stock_data.pct_change().dropna()
    
    # 3. Calculate Portfolio Daily Returns
    # Matrix multiplication of returns and weights
    portfolio_returns = returns.dot(w)
    
    # 4. Annualization Factor
    ann_factor = 252 
    
    # 5. Basic Performance Metrics
    total_return = (1 + portfolio_returns).prod() - 1
    ann_return = portfolio_returns.mean() * ann_factor
    volatility = portfolio_returns.std() * np.sqrt(ann_factor)
    
    # 6. Risk-Adjusted Ratios
    # Sharpe Ratio (using annual metrics)
    excess_return = ann_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility != 0 else 0
    
    # Sortino Ratio (focuses on downside volatility)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(ann_factor)
    sortino_ratio = excess_return / downside_vol if downside_vol != 0 else 0
    
    # 7. Drawdown Analysis
    max_drawdown, _ = calculate_maximum_drawdown(portfolio_returns)
    
    # Calmar Ratio
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 8. Value at Risk (VaR) - Historical Simulation Method
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    
    # 9. Expected Shortfall (ES) - Conditional VaR
    es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    # 10. Statistical Distribution Metrics
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    
    # 11. Beta and Alpha (if benchmark provided)
    metrics = {
        'Annual Return': ann_return,
        'Cumulative Return': total_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'VaR 95%': var_95,
        'VaR 99%': var_99,
        'ES 95%': es_95,
        'ES 99%': es_99,
        'Skewness': skewness,
        'Kurtosis': kurtosis
    }
    
    # Benchmark specific calculations
    if benchmark_returns is not None:
        # Align returns with benchmark
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 2:
            p_ret = portfolio_returns.loc[common_idx]
            b_ret = benchmark_returns.loc[common_idx]
            
            # Beta calculation (Covariance / Variance)
            covariance = np.cov(p_ret, b_ret)[0][1]
            benchmark_variance = np.var(b_ret)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            
            # Alpha calculation (Jensen's Alpha)
            alpha = ann_return - (risk_free_rate + beta * (b_ret.mean() * ann_factor - risk_free_rate))
            
            # Information Ratio
            tracking_error = (p_ret - b_ret).std() * np.sqrt(ann_factor)
            info_ratio = (ann_return - (b_ret.mean() * ann_factor)) / tracking_error if tracking_error != 0 else 0
            
            metrics.update({
                'Beta': beta,
                'Alpha': alpha,
                'Information Ratio': info_ratio
            })

    return metrics, portfolio_returns

def calculate_maximum_drawdown(returns):
    """
    Calculate the Maximum Drawdown and Drawdown series.
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown, drawdown

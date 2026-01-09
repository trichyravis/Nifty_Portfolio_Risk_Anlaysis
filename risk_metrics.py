
"""
Risk Metrics with correct key names matching app.py expectations
Uses 'Annual Return' format instead of 'annual_return'
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calculate_portfolio_metrics(stock_data, weights, market_benchmark, risk_free_rate):
    """
    Calculate portfolio metrics with CORRECT key names for app.py
    
    Returns metrics with keys like: 'Annual Return', 'Sharpe Ratio', etc.
    This matches what your app.py expects
    """
    
    print("\n" + "█"*70)
    print("PORTFOLIO METRICS CALCULATION - WITH CORRECT KEY NAMES")
    print("█"*70)
    
    # ========== INPUT VALIDATION ==========
    
    print(f"\n[1] INPUT VALIDATION")
    print(f"   stock_data type: {type(stock_data).__name__}")
    print(f"   stock_data shape: {stock_data.shape if hasattr(stock_data, 'shape') else 'N/A'}")
    
    if stock_data is None or (hasattr(stock_data, 'empty') and stock_data.empty):
        raise ValueError("stock_data is None or empty")
    
    # ========== ENSURE CORRECT TYPES ==========
    
    print(f"\n[2] TYPE CONVERSION")
    
    if isinstance(stock_data, np.ndarray):
        stock_data = pd.DataFrame(stock_data)
    
    if isinstance(market_benchmark, np.ndarray):
        market_benchmark = pd.Series(market_benchmark)
    elif isinstance(market_benchmark, pd.DataFrame):
        market_benchmark = market_benchmark.iloc[:, 0]
    
    print(f"   stock_data type: {type(stock_data).__name__} ✓")
    print(f"   market_benchmark type: {type(market_benchmark).__name__} ✓")
    
    # ========== CLEAN DATA ==========
    
    print(f"\n[3] DATA CLEANING")
    
    stock_data = stock_data.dropna()
    market_benchmark = market_benchmark.dropna()
    
    print(f"   stock_data: {stock_data.shape}")
    print(f"   benchmark: {len(market_benchmark)}")
    
    # ========== ALIGN INDICES ==========
    
    print(f"\n[4] INDEX ALIGNMENT")
    
    common_index = stock_data.index.intersection(market_benchmark.index)
    
    if len(common_index) < 2:
        raise ValueError(f"Only {len(common_index)} common dates")
    
    stock_data = stock_data.loc[common_index]
    market_benchmark = market_benchmark.loc[common_index]
    
    print(f"   Common dates: {len(common_index)}")
    
    # ========== CALCULATE PORTFOLIO RETURNS ==========
    
    print(f"\n[5] PORTFOLIO RETURNS CALCULATION")
    
    weights = np.asarray(weights).flatten()
    
    if len(weights) != stock_data.shape[1]:
        raise ValueError(f"Weight mismatch: {len(weights)} vs {stock_data.shape[1]}")
    
    if not np.isclose(weights.sum(), 1.0, atol=0.001):
        weights = weights / weights.sum()
    
    portfolio_returns = (stock_data * weights).sum(axis=1).dropna()
    
    print(f"   portfolio_returns: {portfolio_returns.shape}")
    
    # ========== EXTRACT AND FLATTEN ARRAYS ==========
    
    print(f"\n[6] EXTRACT AND FLATTEN ARRAYS")
    
    b_temp = market_benchmark.loc[portfolio_returns.index]
    
    p_ret = np.asarray(portfolio_returns.values).flatten()
    b_ret = np.asarray(b_temp.values).flatten()
    
    print(f"   p_ret: {p_ret.shape}, ndim: {p_ret.ndim}")
    print(f"   b_ret: {b_ret.shape}, ndim: {b_ret.ndim}")
    
    # ========== VERIFY DATA QUALITY ==========
    
    print(f"\n[7] VERIFY DATA QUALITY")
    
    if np.isnan(p_ret).any() or np.isnan(b_ret).any():
        mask = ~(np.isnan(p_ret) | np.isnan(b_ret))
        p_ret = p_ret[mask]
        b_ret = b_ret[mask]
    
    assert p_ret.ndim == 1
    assert len(p_ret) == len(b_ret)
    assert len(p_ret) >= 2
    
    print(f"   ✓ All checks passed")
    
    # ========== CALCULATIONS ==========
    
    print(f"\n[8] RISK METRICS CALCULATIONS")
    
    returns_matrix = np.vstack([p_ret, b_ret])
    covariance = np.cov(returns_matrix)[0, 1]
    
    annual_return = p_ret.mean() * 252
    benchmark_return = b_ret.mean() * 252
    
    annual_volatility = np.std(p_ret, ddof=1) * np.sqrt(252)
    benchmark_volatility = np.std(b_ret, ddof=1) * np.sqrt(252)
    
    benchmark_variance = np.var(b_ret, ddof=1)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    daily_rf = risk_free_rate / 252
    excess_ret = p_ret.mean() - daily_rf
    
    sharpe = (excess_ret * 252) / annual_volatility if annual_volatility > 0 else 0
    
    downside_ret = p_ret[p_ret < daily_rf]
    if len(downside_ret) > 1:
        downside_std = np.std(downside_ret, ddof=1) * np.sqrt(252)
        sortino = (excess_ret * 252) / downside_std if downside_std > 0 else 0
    else:
        downside_std = 0
        sortino = 0
    
    cum = np.cumprod(1 + p_ret)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = np.min(dd)
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # ========== COMPILE RESULTS WITH CORRECT KEY NAMES ==========
    # THIS IS THE KEY DIFFERENCE - Using Title Case with Spaces
    
    metrics = {
        'Annual Return': float(annual_return),
        'Annual Volatility': float(annual_volatility),
        'Sharpe Ratio': float(sharpe),
        'Sortino Ratio': float(sortino),
        'Calmar Ratio': float(calmar),
        'Beta': float(beta),
        'Max Drawdown': float(max_dd),
        'Downside Deviation': float(downside_std),
        'Benchmark Return': float(benchmark_return),
        'Benchmark Volatility': float(benchmark_volatility),
    }
    
    print(f"\n[9] METRICS COMPILED")
    print(f"   Keys: {list(metrics.keys())}")
    
    print(f"\n" + "█"*70)
    print("✓ CALCULATION SUCCESSFUL!")
    print("█"*70 + "\n")
    
    return metrics, portfolio_returns


def safe_wrapper(stock_data, weights, market_benchmark, risk_free_rate):
    """Wrapper with error handling for Streamlit"""
    try:
        metrics, returns = calculate_portfolio_metrics(
            stock_data, weights, market_benchmark, risk_free_rate
        )
        return metrics, returns, None
    except Exception as e:
        import traceback
        error = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error)
        return None, None, error

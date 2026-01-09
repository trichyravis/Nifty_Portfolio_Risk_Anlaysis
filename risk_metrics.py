
"""
ULTIMATE FIX - Handles ALL array dimension issues
With extensive debugging to show exactly what's happening
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calculate_portfolio_metrics(stock_data, weights, market_benchmark, risk_free_rate):
    """
    FINAL WORKING VERSION with all fixes applied
    """
    
    print("\n" + "█"*70)
    print("PORTFOLIO METRICS CALCULATION - ULTRA SAFE VERSION")
    print("█"*70)
    
    # ========== INPUT VALIDATION ==========
    
    print(f"\n[1] INPUT VALIDATION")
    print(f"   stock_data type: {type(stock_data).__name__}")
    print(f"   stock_data shape: {stock_data.shape if hasattr(stock_data, 'shape') else 'N/A'}")
    print(f"   market_benchmark type: {type(market_benchmark).__name__}")
    
    if stock_data is None or (hasattr(stock_data, 'empty') and stock_data.empty):
        raise ValueError("stock_data is None or empty")
    
    # ========== ENSURE CORRECT TYPES ==========
    
    print(f"\n[2] TYPE CONVERSION")
    
    # Ensure stock_data is DataFrame
    if isinstance(stock_data, np.ndarray):
        print(f"   Converting numpy array to DataFrame")
        stock_data = pd.DataFrame(stock_data)
    
    # Ensure benchmark is Series
    if isinstance(market_benchmark, np.ndarray):
        market_benchmark = pd.Series(market_benchmark)
    elif isinstance(market_benchmark, pd.DataFrame):
        # Extract first column if DataFrame
        market_benchmark = market_benchmark.iloc[:, 0]
    
    print(f"   stock_data type: {type(stock_data).__name__} ✓")
    print(f"   market_benchmark type: {type(market_benchmark).__name__} ✓")
    
    # ========== CLEAN DATA ==========
    
    print(f"\n[3] DATA CLEANING")
    print(f"   stock_data before dropna: {stock_data.shape}")
    
    stock_data = stock_data.dropna()
    
    print(f"   stock_data after dropna: {stock_data.shape}")
    print(f"   benchmark before dropna: {len(market_benchmark)}")
    
    market_benchmark = market_benchmark.dropna()
    
    print(f"   benchmark after dropna: {len(market_benchmark)}")
    
    # ========== ALIGN INDICES ==========
    
    print(f"\n[4] INDEX ALIGNMENT")
    
    common_index = stock_data.index.intersection(market_benchmark.index)
    
    print(f"   Common dates: {len(common_index)}")
    
    if len(common_index) < 2:
        raise ValueError(f"Only {len(common_index)} common dates - need at least 2")
    
    stock_data = stock_data.loc[common_index]
    market_benchmark = market_benchmark.loc[common_index]
    
    print(f"   Aligned stock_data: {stock_data.shape}")
    print(f"   Aligned benchmark: {len(market_benchmark)}")
    
    # ========== CALCULATE PORTFOLIO RETURNS ==========
    
    print(f"\n[5] PORTFOLIO RETURNS CALCULATION")
    
    weights = np.asarray(weights).flatten()
    
    print(f"   Weights shape: {weights.shape}")
    print(f"   Stock data columns: {stock_data.shape[1]}")
    
    if len(weights) != stock_data.shape[1]:
        raise ValueError(f"Weight count mismatch: {len(weights)} vs {stock_data.shape[1]}")
    
    if not np.isclose(weights.sum(), 1.0, atol=0.001):
        weights = weights / weights.sum()
        print(f"   Weights normalized ✓")
    
    portfolio_returns = (stock_data * weights).sum(axis=1)
    
    print(f"   portfolio_returns type: {type(portfolio_returns).__name__}")
    print(f"   portfolio_returns shape: {portfolio_returns.shape}")
    print(f"   portfolio_returns.ndim: {portfolio_returns.ndim}")
    
    # Clean
    portfolio_returns = portfolio_returns.dropna()
    
    print(f"   After dropna: {portfolio_returns.shape}")
    
    # ========== EXTRACT ARRAYS - CRITICAL SECTION ==========
    
    print(f"\n[6] EXTRACT AND CONVERT TO NUMPY ARRAYS - CRITICAL!")
    
    # Get benchmark aligned to portfolio
    b_temp = market_benchmark.loc[portfolio_returns.index]
    
    print(f"   benchmark (Series): shape={b_temp.shape}, type={type(b_temp).__name__}")
    
    # METHOD 1: Direct .values
    p_ret = portfolio_returns.values
    b_ret = b_temp.values
    
    print(f"\n   After .values:")
    print(f"   p_ret shape: {p_ret.shape}, ndim: {p_ret.ndim}")
    print(f"   b_ret shape: {b_ret.shape}, ndim: {b_ret.ndim}")
    
    # METHOD 2: Ensure 1D with multiple approaches
    # Try flatten first
    if p_ret.ndim > 1:
        print(f"   p_ret is {p_ret.ndim}D - flattening...")
        p_ret = p_ret.flatten()
    
    if b_ret.ndim > 1:
        print(f"   b_ret is {b_ret.ndim}D - flattening...")
        b_ret = b_ret.flatten()
    
    print(f"\n   After flatten:")
    print(f"   p_ret shape: {p_ret.shape}, ndim: {p_ret.ndim}")
    print(f"   b_ret shape: {b_ret.shape}, ndim: {b_ret.ndim}")
    
    # METHOD 3: Ensure float type
    p_ret = np.asarray(p_ret, dtype=np.float64).flatten()
    b_ret = np.asarray(b_ret, dtype=np.float64).flatten()
    
    print(f"\n   After asarray + flatten:")
    print(f"   p_ret: shape={p_ret.shape}, dtype={p_ret.dtype}")
    print(f"   b_ret: shape={b_ret.shape}, dtype={b_ret.dtype}")
    
    # ========== VERIFY NO NaN ==========
    
    print(f"\n[7] VERIFY DATA QUALITY")
    
    nan_p = np.isnan(p_ret).sum()
    nan_b = np.isnan(b_ret).sum()
    
    print(f"   NaN in p_ret: {nan_p}")
    print(f"   NaN in b_ret: {nan_b}")
    
    if nan_p > 0 or nan_b > 0:
        mask = ~(np.isnan(p_ret) | np.isnan(b_ret))
        p_ret = p_ret[mask]
        b_ret = b_ret[mask]
        print(f"   Removed NaN - new length: {len(p_ret)}")
    
    # ========== FINAL VERIFICATION ==========
    
    print(f"\n[8] FINAL VERIFICATION BEFORE CALCULATIONS")
    
    assert p_ret.ndim == 1, f"p_ret is {p_ret.ndim}D, expected 1D"
    assert b_ret.ndim == 1, f"b_ret is {b_ret.ndim}D, expected 1D"
    assert len(p_ret) == len(b_ret), f"Length mismatch: {len(p_ret)} vs {len(b_ret)}"
    assert len(p_ret) >= 2, f"Insufficient data: {len(p_ret)} points"
    assert not np.isnan(p_ret).any(), "NaN in p_ret"
    assert not np.isnan(b_ret).any(), "NaN in b_ret"
    
    print(f"   ✓ p_ret: 1D array, {len(p_ret)} points")
    print(f"   ✓ b_ret: 1D array, {len(b_ret)} points")
    print(f"   ✓ No NaN values")
    print(f"   ✓ All checks passed!")
    
    # ========== CALCULATIONS ==========
    
    print(f"\n[9] RISK METRICS CALCULATIONS")
    
    # Covariance
    returns_matrix = np.vstack([p_ret, b_ret])
    covariance_matrix = np.cov(returns_matrix)
    covariance = covariance_matrix[0, 1]
    
    # Returns
    annual_return = p_ret.mean() * 252
    benchmark_return = b_ret.mean() * 252
    
    # Volatility
    annual_volatility = np.std(p_ret, ddof=1) * np.sqrt(252)
    benchmark_volatility = np.std(b_ret, ddof=1) * np.sqrt(252)
    
    # Beta
    benchmark_variance = np.var(b_ret, ddof=1)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    # Sharpe Ratio
    daily_rf = risk_free_rate / 252
    excess_ret = p_ret.mean() - daily_rf
    sharpe = (excess_ret * 252) / annual_volatility if annual_volatility > 0 else 0
    
    # Sortino Ratio
    downside_ret = p_ret[p_ret < daily_rf]
    if len(downside_ret) > 1:
        downside_std = np.std(downside_ret, ddof=1) * np.sqrt(252)
        sortino = (excess_ret * 252) / downside_std if downside_std > 0 else 0
    else:
        downside_std = 0
        sortino = 0
    
    # Calmar Ratio
    cum = np.cumprod(1 + p_ret)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = np.min(dd)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    print(f"   Annual return: {annual_return:.4f}")
    print(f"   Volatility: {annual_volatility:.4f}")
    print(f"   Beta: {beta:.4f}")
    print(f"   Sharpe: {sharpe:.4f}")
    print(f"   Sortino: {sortino:.4f}")
    print(f"   Calmar: {calmar:.4f}")
    
    # ========== COMPILE RESULTS ==========
    
    metrics = {
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'beta': float(beta),
        'max_drawdown': float(max_dd),
        'downside_deviation': float(downside_std),
        'benchmark_return': float(benchmark_return),
        'benchmark_volatility': float(benchmark_volatility),
    }
    
    print(f"\n" + "█"*70)
    print("✓ CALCULATION SUCCESSFUL!")
    print("█"*70 + "\n")
    
    return metrics, portfolio_returns


def safe_wrapper(stock_data, weights, market_benchmark, risk_free_rate):
    """Use this in your Streamlit app"""
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


"""
Fixed Risk Metrics Calculation for Nifty Portfolio Analysis
Handles covariance calculations with proper error checking
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_portfolio_metrics(stock_data, weights, market_benchmark, risk_free_rate):
    """
    Calculate portfolio risk metrics (Sharpe, Sortino, Beta, etc.)
    
    Parameters:
    -----------
    stock_data : DataFrame
        Daily returns for each stock
    weights : array
        Portfolio weights
    market_benchmark : Series
        Benchmark returns (same length as stock_data)
    risk_free_rate : float
        Annual risk-free rate (e.g., 0.04 for 4%)
    
    Returns:
    --------
    metrics : dict
        Portfolio metrics (Sharpe, Sortino, Beta, etc.)
    portfolio_returns : Series
        Daily portfolio returns
    """
    
    # ========== STEP 1: VALIDATE INPUT DATA ==========
    
    # Check if stock_data is empty
    if stock_data.empty or len(stock_data) == 0:
        raise ValueError("Stock data is empty")
    
    # Remove NaN values from stock_data
    stock_data_clean = stock_data.dropna()
    
    if len(stock_data_clean) == 0:
        raise ValueError("All stock data values are NaN")
    
    print(f"Stock data shape before cleaning: {stock_data.shape}")
    print(f"Stock data shape after cleaning: {stock_data_clean.shape}")
    
    # ========== STEP 2: ALIGN BENCHMARK WITH PORTFOLIO ==========
    
    # Ensure benchmark is Series and clean
    if not isinstance(market_benchmark, pd.Series):
        market_benchmark = pd.Series(market_benchmark)
    
    # Remove NaN from benchmark
    benchmark_clean = market_benchmark.dropna()
    
    print(f"Benchmark length before cleaning: {len(market_benchmark)}")
    print(f"Benchmark length after cleaning: {len(benchmark_clean)}")
    
    # CRITICAL: Align indices so both have same length
    # Get common dates between stock data and benchmark
    common_index = stock_data_clean.index.intersection(benchmark_clean.index)
    
    if len(common_index) == 0:
        print("WARNING: No common dates between stock data and benchmark")
        print(f"Stock data index range: {stock_data_clean.index[0]} to {stock_data_clean.index[-1]}")
        print(f"Benchmark index range: {benchmark_clean.index[0]} to {benchmark_clean.index[-1]}")
        raise ValueError("Stock data and benchmark have no overlapping dates")
    
    print(f"Common dates available: {len(common_index)}")
    
    # Align both to common index
    stock_data_aligned = stock_data_clean.loc[common_index]
    benchmark_aligned = benchmark_clean.loc[common_index]
    
    # Verify alignment
    assert len(stock_data_aligned) == len(benchmark_aligned), \
        f"Lengths don't match: {len(stock_data_aligned)} vs {len(benchmark_aligned)}"
    
    # ========== STEP 3: CALCULATE PORTFOLIO RETURNS ==========
    
    # Calculate weighted portfolio returns
    portfolio_returns = (stock_data_aligned * weights).sum(axis=1)
    
    print(f"Portfolio returns shape: {portfolio_returns.shape}")
    print(f"Portfolio returns NaN count: {portfolio_returns.isna().sum()}")
    
    # Remove any NaN from portfolio returns
    portfolio_returns_clean = portfolio_returns.dropna()
    
    # ========== STEP 4: VERIFY ARRAY DIMENSIONS ==========
    
    p_ret = portfolio_returns_clean.values  # Convert to numpy array
    b_ret = benchmark_aligned.loc[portfolio_returns_clean.index].values
    
    print(f"\nArray shapes for covariance:")
    print(f"  Portfolio returns shape: {p_ret.shape}")
    print(f"  Benchmark returns shape: {b_ret.shape}")
    print(f"  Portfolio returns dtype: {p_ret.dtype}")
    print(f"  Benchmark returns dtype: {b_ret.dtype}")
    
    # Ensure both are 1D arrays
    p_ret = np.asarray(p_ret).flatten()
    b_ret = np.asarray(b_ret).flatten()
    
    # Verify no NaN values remain
    assert not np.isnan(p_ret).any(), "Portfolio returns contain NaN"
    assert not np.isnan(b_ret).any(), "Benchmark returns contain NaN"
    
    # Verify equal length
    assert len(p_ret) == len(b_ret), \
        f"Length mismatch: p_ret={len(p_ret)}, b_ret={len(b_ret)}"
    
    # ========== STEP 5: CALCULATE COVARIANCE ==========
    
    # FIXED: Use rowvar=False for proper 2D array handling
    # Stack arrays vertically for covariance calculation
    returns_matrix = np.vstack([p_ret, b_ret])
    
    try:
        covariance_matrix = np.cov(returns_matrix)
        covariance = covariance_matrix[0, 1]
        print(f"Covariance calculated successfully: {covariance}")
    except Exception as e:
        print(f"Covariance calculation failed: {e}")
        print(f"Returns matrix shape: {returns_matrix.shape}")
        raise
    
    # ========== STEP 6: CALCULATE BETA ==========
    
    # Beta = Covariance(Portfolio, Benchmark) / Variance(Benchmark)
    benchmark_variance = np.var(b_ret, ddof=1)  # Use sample variance
    
    if benchmark_variance == 0:
        print("WARNING: Benchmark variance is zero, cannot calculate beta")
        beta = 0
    else:
        beta = covariance / benchmark_variance
    
    # ========== STEP 7: CALCULATE RETURNS & VOLATILITY ==========
    
    # Annualize returns (assuming 252 trading days)
    annual_return = p_ret.mean() * 252
    
    # Annualize volatility
    annual_volatility = np.std(p_ret, ddof=1) * np.sqrt(252)
    
    # Benchmark metrics
    benchmark_return = b_ret.mean() * 252
    benchmark_volatility = np.std(b_ret, ddof=1) * np.sqrt(252)
    
    print(f"\nReturns & Volatility:")
    print(f"  Portfolio annual return: {annual_return:.4f}")
    print(f"  Portfolio volatility: {annual_volatility:.4f}")
    print(f"  Benchmark annual return: {benchmark_return:.4f}")
    print(f"  Benchmark volatility: {benchmark_volatility:.4f}")
    
    # ========== STEP 8: CALCULATE SHARPE RATIO ==========
    
    daily_risk_free_rate = risk_free_rate / 252
    excess_return = p_ret.mean() - daily_risk_free_rate
    
    if annual_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = excess_return / (annual_volatility / 252)
    
    # ========== STEP 9: CALCULATE SORTINO RATIO ==========
    
    # Downside deviation (only negative returns)
    downside_returns = p_ret[p_ret < daily_risk_free_rate]
    
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252)
    else:
        downside_std = 0
    
    if downside_std == 0:
        sortino_ratio = 0
    else:
        sortino_ratio = excess_return / (downside_std / 252)
    
    # ========== STEP 10: CALCULATE CALMAR RATIO ==========
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + p_ret)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    if max_drawdown == 0:
        calmar_ratio = 0
    else:
        calmar_ratio = annual_return / abs(max_drawdown)
    
    # ========== STEP 11: CALCULATE INFORMATION RATIO ==========
    
    # Alpha = Portfolio Return - Expected Return (using CAPM)
    expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
    alpha = annual_return - expected_return
    
    # Tracking error
    tracking_error = np.std(p_ret - b_ret, ddof=1) * np.sqrt(252)
    
    if tracking_error == 0:
        information_ratio = 0
    else:
        information_ratio = alpha / tracking_error
    
    # ========== COMPILE RESULTS ==========
    
    metrics = {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'beta': beta,
        'alpha': alpha,
        'information_ratio': information_ratio,
        'max_drawdown': max_drawdown,
        'downside_deviation': downside_std,
        'benchmark_return': benchmark_return,
        'benchmark_volatility': benchmark_volatility,
        'tracking_error': tracking_error,
    }
    
    return metrics, portfolio_returns_clean


def validate_inputs(stock_data, weights, market_benchmark):
    """
    Validate input data before calculations
    
    Parameters:
    -----------
    stock_data : DataFrame
        Stock returns
    weights : array
        Portfolio weights
    market_benchmark : Series/array
        Benchmark returns
    
    Returns:
    --------
    bool : True if valid, raises exception otherwise
    """
    
    # Check stock_data
    if stock_data is None or len(stock_data) == 0:
        raise ValueError("Stock data cannot be empty")
    
    if stock_data.isna().all().all():
        raise ValueError("All stock data values are NaN")
    
    # Check weights
    if weights is None or len(weights) == 0:
        raise ValueError("Weights cannot be empty")
    
    if not np.isclose(sum(weights), 1.0, atol=0.001):
        print(f"WARNING: Weights sum to {sum(weights)}, not 1.0. Normalizing...")
        weights = weights / sum(weights)
    
    # Check benchmark
    if market_benchmark is None or len(market_benchmark) == 0:
        raise ValueError("Benchmark cannot be empty")
    
    if pd.Series(market_benchmark).isna().all():
        raise ValueError("All benchmark values are NaN")
    
    return True


# ============================================================================
# USAGE EXAMPLE FOR STREAMLIT APP
# ============================================================================

def safe_calculate_portfolio_metrics(stock_data, weights, market_benchmark, risk_free_rate):
    """
    Wrapper function with error handling for Streamlit
    """
    
    try:
        # Validate inputs first
        validate_inputs(stock_data, weights, market_benchmark)
        
        # Calculate metrics
        metrics, portfolio_returns = calculate_portfolio_metrics(
            stock_data,
            weights,
            market_benchmark,
            risk_free_rate
        )
        
        return metrics, portfolio_returns, None  # metrics, returns, error
        
    except Exception as e:
        print(f"\nERROR in calculate_portfolio_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, str(e)  # metrics, returns, error message


"""
AUTO-DETECT RISK METRICS - Handles both price data and return data
Includes diagnostics to identify the problem
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def auto_detect_returns(data):
    """
    Auto-detect if data is prices or returns
    Convert prices to returns if needed
    """
    if data is None or data.empty:
        raise ValueError("Data is empty")
    
    # Get first few values to check
    sample = data.iloc[:5] if hasattr(data, 'iloc') else data[:5]
    
    # Check if values look like prices (larger) or returns (smaller)
    mean_val = float(np.abs(data).mean().mean()) if hasattr(data, 'mean') and hasattr(data.mean(), 'mean') else float(np.abs(data).mean())
    
    print(f"[DEBUG] Mean absolute value: {mean_val:.6f}")
    
    # If mean > 1, likely prices, not returns
    if mean_val > 1.0:
        print(f"[DEBUG] Detected PRICE data (mean={mean_val:.4f})")
        print(f"[DEBUG] Converting to daily returns...")
        # Convert prices to returns using pct_change
        returns = data.pct_change().dropna()
        print(f"[DEBUG] Returns mean: {returns.mean().mean() if hasattr(returns, 'mean') and hasattr(returns.mean(), 'mean') else returns.mean():.6f}")
        return returns, "PRICES (converted to returns)"
    else:
        print(f"[DEBUG] Detected RETURN data (mean={mean_val:.6f})")
        return data, "RETURNS (already returns)"


def calculate_portfolio_metrics(stock_data, weights, market_benchmark, risk_free_rate):
    """
    Complete metrics with auto-detection of input data type
    """
    
    print("\n" + "█"*70)
    print("COMPLETE PORTFOLIO METRICS - AUTO-DETECT VERSION")
    print("█"*70)
    
    # ========== AUTO-DETECT INPUT TYPE ==========
    print(f"\n[1] AUTO-DETECTING INPUT DATA TYPE")
    
    if isinstance(stock_data, np.ndarray):
        stock_data = pd.DataFrame(stock_data)
    if isinstance(market_benchmark, np.ndarray):
        market_benchmark = pd.Series(market_benchmark)
    elif isinstance(market_benchmark, pd.DataFrame):
        market_benchmark = market_benchmark.iloc[:, 0]
    
    print(f"   stock_data shape: {stock_data.shape}")
    print(f"   benchmark length: {len(market_benchmark)}")
    
    # Auto-detect and convert
    stock_data_returns, stock_type = auto_detect_returns(stock_data)
    print(f"   Stock data type: {stock_type}")
    
    benchmark_returns, bench_type = auto_detect_returns(pd.DataFrame(market_benchmark))
    benchmark_returns = benchmark_returns.iloc[:, 0]
    print(f"   Benchmark type: {bench_type}")
    
    # ========== CLEAN & ALIGN ==========
    print(f"\n[2] CLEANING AND ALIGNING DATA")
    
    stock_data_returns = stock_data_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    
    common_index = stock_data_returns.index.intersection(benchmark_returns.index)
    if len(common_index) < 2:
        raise ValueError(f"Only {len(common_index)} common dates")
    
    stock_data_returns = stock_data_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    print(f"   Common dates: {len(common_index)}")
    print(f"   Stock returns shape: {stock_data_returns.shape}")
    
    # ========== PORTFOLIO RETURNS ==========
    print(f"\n[3] CALCULATING PORTFOLIO RETURNS")
    
    weights = np.asarray(weights).flatten()
    if len(weights) != stock_data_returns.shape[1]:
        raise ValueError(f"Weight mismatch: {len(weights)} vs {stock_data_returns.shape[1]}")
    
    if not np.isclose(weights.sum(), 1.0, atol=0.001):
        weights = weights / weights.sum()
    
    portfolio_returns = (stock_data_returns * weights).sum(axis=1).dropna()
    
    print(f"   Portfolio returns shape: {portfolio_returns.shape}")
    print(f"   Portfolio mean daily return: {portfolio_returns.mean():.6f}")
    print(f"   Portfolio daily vol: {portfolio_returns.std():.6f}")
    
    # ========== EXTRACT & FLATTEN ==========
    print(f"\n[4] EXTRACTING NUMPY ARRAYS")
    
    b_temp = benchmark_returns.loc[portfolio_returns.index]
    p_ret = np.asarray(portfolio_returns.values).flatten()
    b_ret = np.asarray(b_temp.values).flatten()
    
    print(f"   p_ret shape: {p_ret.shape}, dtype: {p_ret.dtype}")
    print(f"   p_ret range: [{p_ret.min():.6f}, {p_ret.max():.6f}]")
    
    if np.isnan(p_ret).any() or np.isnan(b_ret).any():
        mask = ~(np.isnan(p_ret) | np.isnan(b_ret))
        p_ret = p_ret[mask]
        b_ret = b_ret[mask]
    
    # ========== MAIN CALCULATIONS ==========
    print(f"\n[5] CALCULATING RISK METRICS")
    
    # Annual metrics (252 trading days)
    annual_return = p_ret.mean() * 252
    benchmark_return = b_ret.mean() * 252
    annual_volatility = np.std(p_ret, ddof=1) * np.sqrt(252)
    benchmark_volatility = np.std(b_ret, ddof=1) * np.sqrt(252)
    
    print(f"   Annual return: {annual_return:.6f} ({annual_return*100:.4f}%)")
    print(f"   Annual volatility: {annual_volatility:.6f} ({annual_volatility*100:.4f}%)")
    
    # Beta
    returns_matrix = np.vstack([p_ret, b_ret])
    covariance = np.cov(returns_matrix)[0, 1]
    benchmark_variance = np.var(b_ret, ddof=1)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    # Risk-free rate daily
    daily_rf = risk_free_rate / 252
    excess_ret = p_ret.mean() - daily_rf
    
    # Sharpe Ratio
    sharpe = (excess_ret * 252) / annual_volatility if annual_volatility > 0 else 0
    print(f"   Sharpe Ratio: {sharpe:.6f}")
    
    # Sortino Ratio
    downside_ret = p_ret[p_ret < daily_rf]
    if len(downside_ret) > 1:
        downside_std = np.std(downside_ret, ddof=1) * np.sqrt(252)
        sortino = (excess_ret * 252) / downside_std if downside_std > 0 else 0
    else:
        downside_std = 0
        sortino = 0
    
    # Calmar Ratio & Max Drawdown
    cum = np.cumprod(1 + p_ret)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = np.min(dd)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # ========== VALUE AT RISK ==========
    print(f"\n[6] CALCULATING VaR METRICS")
    
    var_95 = np.percentile(p_ret, 5)
    var_99 = np.percentile(p_ret, 1)
    cvar_95 = p_ret[p_ret <= var_95].mean()
    cvar_99 = p_ret[p_ret <= var_99].mean()
    
    print(f"   VaR 95%: {var_95:.6f}")
    print(f"   CVaR 95%: {cvar_95:.6f}")
    
    # ========== ADDITIONAL METRICS ==========
    print(f"\n[7] CALCULATING DISTRIBUTION METRICS")
    
    # Skewness
    mean_ret = p_ret.mean()
    std_ret = p_ret.std()
    skewness = ((p_ret - mean_ret) ** 3).mean() / (std_ret ** 3) if std_ret > 0 else 0
    
    # Kurtosis
    kurtosis = ((p_ret - mean_ret) ** 4).mean() / (std_ret ** 4) - 3 if std_ret > 0 else 0
    
    # Information Ratio
    tracking_error = np.std(p_ret - b_ret, ddof=1) * np.sqrt(252)
    alpha = annual_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    info_ratio = alpha / tracking_error if tracking_error > 0 else 0
    
    # ========== COMPILE METRICS ==========
    print(f"\n[8] COMPILING FINAL METRICS")
    
    metrics = {
        'Annual Return': float(annual_return),
        'Volatility': float(annual_volatility),
        'Sharpe Ratio': float(sharpe),
        'Sortino Ratio': float(sortino),
        'Calmar Ratio': float(calmar),
        'Beta': float(beta),
        'Max Drawdown': float(max_dd),
        'Downside Deviation': float(downside_std),
        'Benchmark Return': float(benchmark_return),
        'Benchmark Volatility': float(benchmark_volatility),
        'VaR 95%': float(var_95),
        'VaR 99%': float(var_99),
        'CVaR 95%': float(cvar_95),
        'CVaR 99%': float(cvar_99),
        'Skewness': float(skewness),
        'Kurtosis': float(kurtosis),
        'Information Ratio': float(info_ratio),
        'Alpha': float(alpha),
    }
    
    print(f"\n[SUMMARY] Final metrics:")
    print(f"   Return: {metrics['Annual Return']:.6f} ({metrics['Annual Return']*100:.4f}%)")
    print(f"   Volatility: {metrics['Volatility']:.6f} ({metrics['Volatility']*100:.4f}%)")
    print(f"   Sharpe: {metrics['Sharpe Ratio']:.6f}")
    print(f"   Max Drawdown: {metrics['Max Drawdown']:.6f} ({metrics['Max Drawdown']*100:.4f}%)")
    print(f"   VaR 95%: {metrics['VaR 95%']:.6f} ({metrics['VaR 95%']*100:.4f}%)")
    
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

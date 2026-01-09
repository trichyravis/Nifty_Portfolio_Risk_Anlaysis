"""
Configuration file for Portfolio Risk Analysis App
The Mountain Path - World of Finance
"""

# App Configuration
APP_CONFIG = {
    'title': 'Portfolio Performance Risk Analysis',
    'subtitle': 'Advanced Risk Metrics & Performance Analytics',
    'author': 'Prof. V. Ravichandran',
    'organization': 'The Mountain Path - World of Finance',
    'version': '1.0.0'
}

# Color Scheme - Mountain Path Theme
COLORS = {
    'primary': '#003366',      # Dark Blue
    'secondary': '#004d80',    # Light Blue  
    'accent': '#FFD700',       # Gold
    'success': '#28a745',      # Green
    'danger': '#dc3545',       # Red
    'warning': '#ffc107',      # Yellow
    'info': '#17a2b8',         # Cyan
    'light': '#f8f9fa',        # Light Gray
    'dark': '#343a40',         # Dark Gray
    'background': '#F0F8FF',   # Alice Blue
    'card_bg': '#FFFFFF'       # White
}

# Typography
FONTS = {
    'primary': 'Arial, sans-serif',
    'secondary': 'Georgia, serif',
    'monospace': 'Courier New, monospace'
}

# Nifty 50 Stocks List (as of 2024)
NIFTY_50_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'AXISBANK.NS': 'Axis Bank',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
    'WIPRO.NS': 'Wipro',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'TITAN.NS': 'Titan Company',
    'NESTLEIND.NS': 'Nestle India',
    'SUNPHARMA.NS': 'Sun Pharmaceutical',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'ADANIENT.NS': 'Adani Enterprises',
    'TATAMOTORS.NS': 'Tata Motors',
    'ONGC.NS': 'ONGC',
    'HCLTECH.NS': 'HCL Technologies',
    'JSWSTEEL.NS': 'JSW Steel',
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC Limited',
    'M&M.NS': 'Mahindra & Mahindra',
    'TATASTEEL.NS': 'Tata Steel',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'TECHM.NS': 'Tech Mahindra',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'HINDALCO.NS': 'Hindalco Industries',
    'GRASIM.NS': 'Grasim Industries',
    'DIVISLAB.NS': 'Divis Laboratories',
    'CIPLA.NS': 'Cipla',
    'DRREDDY.NS': 'Dr. Reddys Laboratories',
    'BRITANNIA.NS': 'Britannia Industries',
    'EICHERMOT.NS': 'Eicher Motors',
    'COALINDIA.NS': 'Coal India',
    'BPCL.NS': 'Bharat Petroleum',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'UPL.NS': 'UPL Limited',
    'TATACONSUM.NS': 'Tata Consumer Products',
    'ADANIGREEN.NS': 'Adani Green Energy',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
    'VEDL.NS': 'Vedanta Limited',
    'SBILIFE.NS': 'SBI Life Insurance',
    'SHREECEM.NS': 'Shree Cement',
    'BAJAJ-AUTO.NS': 'Bajaj Auto'
}

# Risk Parameters
RISK_PARAMS = {
    'confidence_levels': [0.95, 0.99],
    'risk_free_rate': 0.065,  # 6.5% annual
    'benchmark_index': '^NSEI',  # Nifty 50 Index
    'trading_days': 252
}

# Date Periods
DATE_PERIODS = {
    '1M': '1 Month',
    '3M': '3 Months',
    '6M': '6 Months',
    '1Y': '1 Year',
    '2Y': '2 Years',
    '3Y': '3 Years',
    '5Y': '5 Years'
}

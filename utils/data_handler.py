
# In utils/data_handler.py
@staticmethod
@lru_cache(maxsize=128)
def fetch_stock_data(tickers, start_date, end_date):
    try:
        # Explicitly set auto_adjust=False if you MUST have a separate 'Adj Close' column,
        # otherwise use the new default and look for 'Close'.
        raw_data = yf.download(
            list(tickers),
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False # Set to False to ensure 'Adj Close' is generated
        )
        
        # Robustly handle Column Selection
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        else:
            data = raw_data['Close']
            
        # Ensure it remains a DataFrame even for a single ticker
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


"""
Portfolio Performance Risk Analysis Application
The Mountain Path - World of Finance
Author: Prof. V. Ravichandran
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from io import BytesIO

# Import custom modules
from config import *
from styles import load_css
from components import *
from risk_metrics import *

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG['title'],
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Initialize session state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'weights' not in st.session_state:
    st.session_state.weights = {}

# Hero Header
render_hero_header()

# Sidebar Configuration
with st.sidebar:
    st.markdown(f"### ⚙️ Portfolio Configuration")
    
    # Stock Selection
    st.markdown("#### 📈 Select Stocks")
    selected_stocks = st.multiselect(
        "Choose stocks from Nifty 50",
        options=list(NIFTY_50_STOCKS.keys()),
        default=list(NIFTY_50_STOCKS.keys())[:5],
        format_func=lambda x: f"{NIFTY_50_STOCKS[x]} ({x})",
        help="Select multiple stocks for your portfolio"
    )
    
    # Period Selection
    st.markdown("#### 📅 Investment Period")
    period = st.selectbox(
        "Select period",
        options=list(DATE_PERIODS.keys()),
        format_func=lambda x: DATE_PERIODS[x],
        index=3
    )
    
    # Weight Allocation
    if selected_stocks:
        st.markdown("#### ⚖️ Portfolio Weights (%)")
        st.info("Weights must sum to 100%")
        
        weights = {}
        equal_weight = 100 / len(selected_stocks)
        
        for stock in selected_stocks:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(NIFTY_50_STOCKS[stock][:20])
            with col2:
                weights[stock] = st.number_input(
                    f"Weight",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(equal_weight, 2),
                    step=0.1,
                    key=f"weight_{stock}",
                    label_visibility="collapsed"
                )
        
        total_weight = sum(weights.values())
        
        if abs(total_weight - 100) > 0.01:
            st.error(f"⚠️ Total weight: {total_weight:.2f}% (Must be 100%)")
        else:
            st.success(f"✅ Total weight: {total_weight:.2f}%")
    
    # Analyze Button
    st.markdown("---")
    analyze_button = st.button("🔍 Analyze Portfolio", type="primary", use_container_width=True)

# Main Content Area
if analyze_button and selected_stocks and abs(sum(weights.values()) - 100) <= 0.01:
    with st.spinner("📊 Fetching data and calculating metrics..."):
        # Calculate date range
        end_date = datetime.now()
        period_map = {
            '1M': 30, '3M': 90, '6M': 180,
            '1Y': 365, '2Y': 730, '3Y': 1095, '5Y': 1825
        }
        start_date = end_date - timedelta(days=period_map[period])
        
        # Robust Fetch stock data
        raw_stock_download = yf.download(
            selected_stocks,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        if 'Adj Close' in raw_stock_download.columns:
            stock_data = raw_stock_download['Adj Close']
        else:
            stock_data = raw_stock_download['Close']
            
        if len(selected_stocks) == 1:
            stock_data = pd.DataFrame(stock_data)
            stock_data.columns = selected_stocks
        
        # Robust Fetch benchmark data
        raw_bench_download = yf.download(
            RISK_PARAMS['benchmark_index'],
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        if 'Adj Close' in raw_bench_download.columns:
            benchmark_data = raw_bench_download['Adj Close']
        else:
            benchmark_data = raw_bench_download['Close']
        
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        # Convert weights to array
        weight_array = np.array([weights[stock]/100 for stock in selected_stocks])
        
        # Calculate metrics
        metrics, portfolio_returns = calculate_portfolio_metrics(
            stock_data,
            weight_array,
            benchmark_returns,
            RISK_PARAMS['risk_free_rate']
        )
        
        # Store in session state
        st.session_state.portfolio_data = {
            'stock_data': stock_data,
            'metrics': metrics,
            'portfolio_returns': portfolio_returns,
            'weights': weights,
            'selected_stocks': selected_stocks
        }

# Display Results in Tabs
if st.session_state.portfolio_data:
    data = st.session_state.portfolio_data
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "📈 Performance",
        "⚠️ Risk Metrics",
        "📉 Drawdown Analysis",
        "🔄 Correlation",
        "📋 Detailed Report",
        "ℹ️ About"
    ])
    
    with tab1:
        st.markdown("### Portfolio Overview")
        
        # Key Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(render_metric_card(
                "Annual Return",
                f"{data['metrics']['Annual Return']*100:.2f}%",
                delta=data['metrics']['Annual Return']*100,
                delta_color="normal" if data['metrics']['Annual Return'] > 0 else "inverse"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(render_metric_card(
                "Volatility",
                f"{data['metrics']['Volatility']*100:.2f}%"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(render_metric_card(
                "Sharpe Ratio",
                f"{data['metrics']['Sharpe Ratio']:.3f}"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(render_metric_card(
                "Max Drawdown",
                f"{data['metrics']['Max Drawdown']*100:.2f}%"
            ), unsafe_allow_html=True)
        
        # Portfolio Composition
        st.markdown("### Portfolio Composition")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            composition_df = pd.DataFrame({
                'Stock': [NIFTY_50_STOCKS[s] for s in data['selected_stocks']],
                'Ticker': data['selected_stocks'],
                'Weight (%)': [data['weights'][s] for s in data['selected_stocks']]
            })
            st.dataframe(composition_df, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=[NIFTY_50_STOCKS[s] for s in data['selected_stocks']],
                values=[data['weights'][s] for s in data['selected_stocks']],
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig.update_layout(
                title="Portfolio Allocation",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Performance Analysis")
        
        # Cumulative Returns Chart
        cumulative_returns = (1 + data['portfolio_returns']).cumprod()
        cumulative_df = pd.DataFrame({
            'Portfolio': (cumulative_returns - 1) * 100
        }, index=cumulative_returns.index)
        
        # Add individual stock returns
        for stock in data['selected_stocks']:
            stock_returns = data['stock_data'][stock].pct_change().dropna()
            cumulative_df[NIFTY_50_STOCKS[stock]] = ((1 + stock_returns).cumprod() - 1) * 100
        
        fig = create_returns_chart(cumulative_df, "Cumulative Returns Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Metrics Table
        st.markdown("### Performance Ratios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(render_metric_card(
                "Sharpe Ratio",
                f"{data['metrics']['Sharpe Ratio']:.3f}"
            ), unsafe_allow_html=True)
            st.caption("Risk-adjusted return relative to total volatility")
        
        with col2:
            st.markdown(render_metric_card(
                "Sortino Ratio",
                f"{data['metrics']['Sortino Ratio']:.3f}"
            ), unsafe_allow_html=True)
            st.caption("Risk-adjusted return relative to downside volatility")
        
        with col3:
            st.markdown(render_metric_card(
                "Calmar Ratio",
                f"{data['metrics']['Calmar Ratio']:.3f}"
            ), unsafe_allow_html=True)
            st.caption("Annual return relative to maximum drawdown")
    
    with tab3:
        st.markdown("### Risk Metrics Analysis")
        
        # VaR and ES Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Value at Risk (VaR)")
            
            var_95_daily = data['metrics']['VaR 95%'] * 100
            var_99_daily = data['metrics']['VaR 99%'] * 100
            
            st.info(f"""
            **95% Confidence Level** Daily VaR: {var_95_daily:.2f}%  
            Monthly VaR: {var_95_daily * np.sqrt(21):.2f}%  
            
            **99% Confidence Level** Daily VaR: {var_99_daily:.2f}%  
            Monthly VaR: {var_99_daily * np.sqrt(21):.2f}%
            """)
        
        with col2:
            st.markdown("#### Expected Shortfall (CVaR)")
            
            es_95_daily = data['metrics']['ES 95%'] * 100
            es_99_daily = data['metrics']['ES 99%'] * 100
            
            st.warning(f"""
            **95% Confidence Level** Daily ES: {es_95_daily:.2f}%  
            Monthly ES: {es_95_daily * np.sqrt(21):.2f}%  
            
            **99% Confidence Level** Daily ES: {es_99_daily:.2f}%  
            Monthly ES: {es_99_daily * np.sqrt(21):.2f}%
            """)
        
        # Distribution Chart
        fig = create_var_distribution(
            data['portfolio_returns'],
            data['metrics']['VaR 95%'],
            data['metrics']['VaR 99%'],
            data['metrics']['ES 95%'],
            data['metrics']['ES 99%']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional Risk Metrics
        st.markdown("#### Statistical Measures")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Standard Deviation", 
                      f"{data['portfolio_returns'].std()*100:.3f}%")
        
        with col2:
            st.metric("Skewness", 
                      f"{data['metrics'].get('Skewness', 0):.3f}")
        
        with col3:
            st.metric("Kurtosis", 
                      f"{data['metrics'].get('Kurtosis', 0):.3f}")
        
        with col4:
            if 'Beta' in data['metrics']:
                st.metric("Beta", 
                          f"{data['metrics']['Beta']:.3f}")
    
    with tab4:
        st.markdown("### Drawdown Analysis")
        
        # Calculate drawdown series
        _, drawdown_series = calculate_maximum_drawdown(data['portfolio_returns'])
        
        # Drawdown Chart
        fig = create_drawdown_chart(drawdown_series)
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Maximum Drawdown", 
                      f"{data['metrics']['Max Drawdown']*100:.2f}%")
        
        with col2:
            recovery_periods = []
            in_drawdown = False
            count = 0
            
            for dd in drawdown_series:
                if dd < 0:
                    if not in_drawdown:
                        in_drawdown = True
                        count = 1
                    else:
                        count += 1
                else:
                    if in_drawdown:
                        recovery_periods.append(count)
                        in_drawdown = False
                        count = 0
            
            avg_recovery = np.mean(recovery_periods) if recovery_periods else 0
            st.metric("Avg Recovery Period", 
                      f"{avg_recovery:.0f} days")
        
        with col3:
            current_dd = drawdown_series.iloc[-1] * 100
            st.metric("Current Drawdown", 
                      f"{current_dd:.2f}%")
        
        # Drawdown Distribution
        st.markdown("#### Drawdown Distribution")
        
        drawdown_stats = pd.DataFrame({
            'Metric': ['Mean', 'Median', '25th Percentile', '75th Percentile', 'Worst 5%'],
            'Value (%)': [
                drawdown_series.mean() * 100,
                drawdown_series.median() * 100,
                drawdown_series.quantile(0.25) * 100,
                drawdown_series.quantile(0.75) * 100,
                drawdown_series.quantile(0.05) * 100
            ]
        })
        
        st.dataframe(drawdown_stats, use_container_width=True)
    
    with tab5:
        st.markdown("### Correlation Analysis")
        
        # Calculate correlation matrix
        returns_df = data['stock_data'].pct_change().dropna()
        correlation_matrix = returns_df.corr()
        
        # Rename columns for display
        display_corr = correlation_matrix.copy()
        display_corr.index = [NIFTY_50_STOCKS[s] for s in data['selected_stocks']]
        display_corr.columns = [NIFTY_50_STOCKS[s] for s in data['selected_stocks']]
        
        # Correlation Heatmap
        fig = create_correlation_heatmap(display_corr)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Statistics
        st.markdown("#### Correlation Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average correlation
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            avg_corr = correlation_matrix.where(mask).stack().mean()
            
            st.info(f"""
            **Portfolio Diversification Metrics** Average Pairwise Correlation: {avg_corr:.3f}  
            Max Correlation: {correlation_matrix.where(mask).max().max():.3f}  
            Min Correlation: {correlation_matrix.where(mask).min().min():.3f}
            """)
        
        with col2:
            # Most/Least correlated pairs
            corr_pairs = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_pairs.append({
                        'Pair': f"{NIFTY_50_STOCKS[data['selected_stocks'][i]]} - {NIFTY_50_STOCKS[data['selected_stocks'][j]]}",
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
            
            if corr_pairs:
                corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                st.warning(f"""
                **Correlation Pairs** Highest: {corr_pairs_df.iloc[0]['Pair']}  
                ({corr_pairs_df.iloc[0]['Correlation']:.3f})  
                
                Lowest: {corr_pairs_df.iloc[-1]['Pair']}  
                ({corr_pairs_df.iloc[-1]['Correlation']:.3f})
                """)
    
    with tab6:
        st.markdown("### Detailed Portfolio Report")
        
        # Summary Section
        st.markdown("#### Executive Summary")
        
        summary_text = f"""
        This portfolio consists of {len(data['selected_stocks'])} stocks from the Nifty 50 index, 
        analyzed over the {DATE_PERIODS[period]} period. The portfolio generated an annual return of 
        {data['metrics']['Annual Return']*100:.2f}% with a volatility of {data['metrics']['Volatility']*100:.2f}%.
        
        The risk-adjusted performance, measured by the Sharpe Ratio of {data['metrics']['Sharpe Ratio']:.3f}, 
        indicates {'positive' if data['metrics']['Sharpe Ratio'] > 0 else 'negative'} risk-adjusted returns. 
        The maximum drawdown of {abs(data['metrics']['Max Drawdown'])*100:.2f}% represents the largest 
        peak-to-trough decline during the investment period.
        """
        
        st.write(summary_text)
        
        # Detailed Metrics Table
        st.markdown("#### Complete Risk-Return Metrics")
        
        metrics_list = [
            {'Metric': 'Annual Return', 'Value': f"{data['metrics']['Annual Return']*100:.2f}%", 'Category': 'Return'},
            {'Metric': 'Volatility (Annual)', 'Value': f"{data['metrics']['Volatility']*100:.2f}%", 'Category': 'Risk'},
            {'Metric': 'Sharpe Ratio', 'Value': f"{data['metrics']['Sharpe Ratio']:.3f}", 'Category': 'Risk-Adjusted'},
            {'Metric': 'Sortino Ratio', 'Value': f"{data['metrics']['Sortino Ratio']:.3f}", 'Category': 'Risk-Adjusted'},
            {'Metric': 'Calmar Ratio', 'Value': f"{data['metrics']['Calmar Ratio']:.3f}", 'Category': 'Risk-Adjusted'},
            {'Metric': 'Value at Risk (95%)', 'Value': f"{data['metrics']['VaR 95%']*100:.2f}%", 'Category': 'Risk'},
            {'Metric': 'Value at Risk (99%)', 'Value': f"{data['metrics']['VaR 99%']*100:.2f}%", 'Category': 'Risk'},
            {'Metric': 'Expected Shortfall (95%)', 'Value': f"{data['metrics']['ES 95%']*100:.2f}%", 'Category': 'Risk'},
            {'Metric': 'Expected Shortfall (99%)', 'Value': f"{data['metrics']['ES 99%']*100:.2f}%", 'Category': 'Risk'},
            {'Metric': 'Maximum Drawdown', 'Value': f"{data['metrics']['Max Drawdown']*100:.2f}%", 'Category': 'Risk'},
            {'Metric': 'Skewness', 'Value': f"{data['metrics'].get('Skewness', 0):.3f}", 'Category': 'Distribution'},
            {'Metric': 'Kurtosis', 'Value': f"{data['metrics'].get('Kurtosis', 0):.3f}", 'Category': 'Distribution'},
        ]
        
        if 'Information Ratio' in data['metrics']:
            metrics_list.extend([
                {'Metric': 'Information Ratio', 'Value': f"{data['metrics']['Information Ratio']:.3f}", 'Category': 'Risk-Adjusted'},
                {'Metric': 'Beta', 'Value': f"{data['metrics'].get('Beta', 0):.3f}", 'Category': 'Market Risk'},
                {'Metric': 'Alpha', 'Value': f"{data['metrics'].get('Alpha', 0)*100:.2f}%", 'Category': 'Return'}
            ])
            
        metrics_df = pd.DataFrame(metrics_list)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Download Report
        st.markdown("#### Export Report")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            composition_df.to_excel(writer, sheet_name='Portfolio Composition', index=False)
            metrics_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
            display_corr.to_excel(writer, sheet_name='Correlation Matrix')
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Full Report (Excel)",
            data=excel_data,
            file_name=f"portfolio_risk_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with tab7:
        st.markdown("### About This Model")
        
        st.markdown(f"""
        #### 📊 Portfolio Performance Risk Analysis Framework
        
        This application provides comprehensive risk analysis for equity portfolios using advanced 
        quantitative metrics and modern portfolio theory principles.
        
        ##### Key Features:
        
        **1. Risk Metrics:**
        - **Value at Risk (VaR)**: Statistical measure of potential loss at specified confidence levels
        - **Expected Shortfall (CVaR)**: Average loss beyond VaR threshold
        - **Maximum Drawdown**: Largest peak-to-trough portfolio decline
        - **Volatility**: Standard deviation of returns (annualized)
        
        **2. Performance Ratios:**
        - **Sharpe Ratio**: Risk-adjusted return relative to total risk
        - **Sortino Ratio**: Risk-adjusted return focusing on downside risk
        - **Calmar Ratio**: Return relative to maximum drawdown
        - **Information Ratio**: Active return relative to tracking error
        
        **3. Statistical Measures:**
        - **Skewness**: Asymmetry of return distribution
        - **Kurtosis**: Tail risk in return distribution
        - **Beta**: Systematic risk relative to market
        - **Alpha**: Excess return above market expectation
        
        ##### Methodology:
        
        1. **Data Source**: Real-time stock prices from Yahoo Finance
        2. **Return Calculation**: Simple returns based on price change
        3. **Risk Measures**: Historical simulation approach
        4. **Confidence Levels**: 95% and 99% for VaR and ES
        5. **Time Horizon**: Scalable from daily to annual metrics
        
        ---
        
        **Disclaimer**: This tool is for educational and research purposes only. Past performance 
        does not guarantee future results. Always consult with qualified financial advisors for 
        investment decisions.
        
        ---
        
        **Author**: Prof. V. Ravichandran  
        **Experience**: 28+ Years Corporate Finance & Banking, 10+ Years Academic Excellence  
        **Platform**: The Mountain Path - World of Finance  
        **Version**: 1.0.0  
        **Last Updated**: {datetime.now().strftime('%B %Y')}
        """)
        
        # Risk Warning
        render_info_box(
            "Risk Disclosure",
            """
            All investments carry risk. The value of investments can fall as well as rise, and you may get back 
            less than you invest. The risk metrics presented are based on historical data and may not predict 
            future performance accurately. This tool should not be considered as personal investment advice.
            """,
            box_type="warning"
        )

else:
    # Instructions when no analysis is run
    st.markdown("### 📌 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Step 1: Select Stocks** Choose stocks from the Nifty 50 index using the sidebar
        """)
    
    with col2:
        st.info("""
        **Step 2: Set Weights** Allocate portfolio weights (must sum to 100%)
        """)
    
    with col3:
        st.info("""
        **Step 3: Analyze** Click 'Analyze Portfolio' to generate risk metrics
        """)
    
    # Display sample metrics
    st.markdown("### 📊 Available Risk Metrics")
    
    metrics_info = {
        "Value at Risk (VaR)": "Maximum expected loss at a given confidence level",
        "Expected Shortfall": "Average loss beyond the VaR threshold",
        "Maximum Drawdown": "Largest peak-to-trough decline in portfolio value",
        "Sharpe Ratio": "Risk-adjusted return per unit of total risk",
        "Sortino Ratio": "Risk-adjusted return per unit of downside risk",
        "Calmar Ratio": "Annual return divided by maximum drawdown",
        "Information Ratio": "Excess return relative to tracking error"
    }
    
    for metric, description in metrics_info.items():
        st.markdown(f"**{metric}**: {description}")

# Footer
render_footer()

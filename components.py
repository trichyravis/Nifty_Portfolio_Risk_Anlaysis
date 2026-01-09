"""
Reusable UI Components for Portfolio Risk Analysis App
The Mountain Path - World of Finance
"""

import streamlit as st
from config import COLORS, APP_CONFIG
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_hero_header():
    """Render the hero header section"""
    st.markdown(f"""
        <div class="hero-header">
            <h1 class="hero-title">📊 {APP_CONFIG['title']}</h1>
            <p class="hero-subtitle">{APP_CONFIG['subtitle']}</p>
            <p style="text-align: center; color: {COLORS['light']}; font-size: 0.9rem;">
                {APP_CONFIG['author']} | {APP_CONFIG['organization']}
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, delta_color=None):
    """Render a metric card"""
    delta_html = ""
    if delta is not None:
        delta_class = "metric-positive" if delta_color == "normal" else "metric-negative"
        delta_symbol = "↑" if delta > 0 else "↓"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {abs(delta):.2f}%</div>'
    
    return f"""
        <div class="risk-metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """

def render_info_box(title, content, box_type="info"):
    """Render an information box"""
    box_class = "info-box" if box_type == "info" else "warning-box"
    icon = "ℹ️" if box_type == "info" else "⚠️"
    
    st.markdown(f"""
        <div class="{box_class}">
            <h4>{icon} {title}</h4>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the footer section"""
    st.markdown(f"""
        <div class="footer">
            <p class="footer-text">© 2024 {APP_CONFIG['organization']}</p>
            <p class="footer-text">
                Developed by {APP_CONFIG['author']} | 
                28+ Years Corporate Finance & Banking Experience | 
                10+ Years Academic Excellence
            </p>
            <p>
                <a href="#" class="footer-link">Documentation</a> |
                <a href="#" class="footer-link">Support</a> |
                <a href="#" class="footer-link">Contact</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

def create_returns_chart(returns_df, title="Portfolio Returns"):
    """Create returns line chart"""
    fig = go.Figure()
    
    for column in returns_df.columns:
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df[column],
            mode='lines',
            name=column,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Returns (%)",
        template="plotly_white",
        height=400,
        hovermode='x unified',
        plot_bgcolor=COLORS['light'],
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=20, color=COLORS['primary']),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_risk_return_scatter(risk_return_df):
    """Create risk-return scatter plot"""
    fig = px.scatter(
        risk_return_df,
        x='Risk (Std Dev)',
        y='Return',
        text='Asset',
        color='Sharpe Ratio',
        size='Weight',
        color_continuous_scale='RdYlGn',
        template='plotly_white'
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title="Risk-Return Profile",
        height=500,
        plot_bgcolor=COLORS['light'],
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=20, color=COLORS['primary'])
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix):
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=500,
        template="plotly_white",
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=20, color=COLORS['primary'])
    )
    
    return fig

def create_drawdown_chart(drawdown_series):
    """Create drawdown chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color=COLORS['danger'], width=2),
        fillcolor=f"rgba(220, 53, 69, 0.3)"
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=400,
        plot_bgcolor=COLORS['light'],
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=20, color=COLORS['primary']),
        yaxis=dict(tickformat='.1f')
    )
    
    return fig

def create_var_distribution(returns, var_95, var_99, es_95, es_99):
    """Create VaR distribution chart"""
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Returns Distribution',
        marker_color=COLORS['secondary'],
        opacity=0.7
    ))
    
    # Add VaR lines
    fig.add_vline(x=var_95 * 100, line_dash="dash", 
                  line_color=COLORS['warning'], 
                  annotation_text="VaR 95%")
    fig.add_vline(x=var_99 * 100, line_dash="dash", 
                  line_color=COLORS['danger'], 
                  annotation_text="VaR 99%")
    
    # Add ES lines
    fig.add_vline(x=es_95 * 100, line_dash="solid", 
                  line_color=COLORS['warning'], 
                  annotation_text="ES 95%", opacity=0.5)
    fig.add_vline(x=es_99 * 100, line_dash="solid", 
                  line_color=COLORS['danger'], 
                  annotation_text="ES 99%", opacity=0.5)
    
    fig.update_layout(
        title="Returns Distribution with VaR & Expected Shortfall",
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        plot_bgcolor=COLORS['light'],
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=20, color=COLORS['primary']),
        showlegend=True
    )
    
    return fig

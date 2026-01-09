
"""
CSS Styles for Portfolio Risk Analysis App
The Mountain Path - World of Finance
"""

from config import COLORS, FONTS

def load_css():
    """Load custom CSS styles"""
    return f"""
    <style>
    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, {COLORS['light']} 100%);
        font-family: {FONTS['primary']};
    }}
    
    /* Hero Header */
    .hero-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .hero-title {{
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: white;
        text-align: center;
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        color: {COLORS['accent']};
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    /* Cards */
    .metric-card {{
        background: {COLORS['card_bg']};
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    .risk-metric-card {{
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 3px solid {COLORS['accent']};
        margin-bottom: 1rem;
    }}
    
    /* Metrics Display */
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {COLORS['primary']};
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {COLORS['dark']};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .metric-delta {{
        font-size: 1rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
        margin-top: 0.5rem;
    }}
    
    .metric-positive {{
        color: {COLORS['success']};
        background: rgba(40, 167, 69, 0.1);
    }}
    
    .metric-negative {{
        color: {COLORS['danger']};
        background: rgba(220, 53, 69, 0.1);
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
    }}
    
    /* Info Box */
    .info-box {{
        background: {COLORS['light']};
        border-left: 4px solid {COLORS['info']};
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid {COLORS['warning']};
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }}
    
    /* Tables */
    .dataframe {{
        font-size: 0.9rem;
        border-collapse: collapse;
        width: 100%;
    }}
    
    .dataframe th {{
        background: {COLORS['primary']};
        color: white;
        padding: 0.75rem;
        text-align: left;
        font-weight: bold;
    }}
    
    .dataframe td {{
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }}
    
    .dataframe tr:hover {{
        background: {COLORS['light']};
    }}
    
    /* Footer */
    .footer {{
        background: {COLORS['dark']};
        color: white;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        border-radius: 10px 10px 0 0;
    }}
    
    .footer-text {{
        color: {COLORS['light']};
        margin-bottom: 0.5rem;
    }}
    
    .footer-link {{
        color: {COLORS['accent']};
        text-decoration: none;
        margin: 0 0.5rem;
    }}
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: {COLORS['card_bg']};
        padding: 1rem;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: {COLORS['light']};
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']};
        color: white;
    }}
    </style>
    """

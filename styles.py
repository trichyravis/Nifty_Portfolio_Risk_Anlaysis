
"""
CSS Styles for Portfolio Risk Analysis App
The Mountain Path - World of Finance
Integrated with Footer and Complete Styling
"""

from config import COLORS, FONTS

def load_css():
    """Load custom CSS styles for the app"""
    return f"""
    <style>
    /* ==================== GLOBAL STYLES ==================== */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, {COLORS['light']} 100%);
        font-family: {FONTS['primary']};
    }}
    
    body {{
        margin: 0;
        padding: 0;
    }}
    
    /* ==================== HERO HEADER ==================== */
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
    
    /* ==================== CARDS & CONTAINERS ==================== */
    .metric-card {{
        background: {COLORS['card_bg']};
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }}
    
    .risk-metric-card {{
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 3px solid {COLORS['accent']};
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}
    
    .risk-metric-card:hover {{
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        border-left-color: {COLORS['primary']};
    }}
    
    /* ==================== METRICS DISPLAY ==================== */
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
        font-weight: bold;
    }}
    
    .metric-positive {{
        color: {COLORS['success']};
        background: rgba(40, 167, 69, 0.1);
    }}
    
    .metric-negative {{
        color: {COLORS['danger']};
        background: rgba(220, 53, 69, 0.1);
    }}
    
    /* ==================== BUTTONS ==================== */
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
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* ==================== SIDEBAR ==================== */
    .css-1d391kg {{
        background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
    }}
    
    .sidebar-section {{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* ==================== INFO BOXES ==================== */
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
    
    .success-box {{
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid {COLORS['success']};
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }}
    
    .danger-box {{
        background: rgba(220, 53, 69, 0.1);
        border-left: 4px solid {COLORS['danger']};
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }}
    
    /* ==================== TABLES ==================== */
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
    
    /* ==================== TABS ==================== */
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
    
    /* ==================== CHARTS & VISUALIZATIONS ==================== */
    .chart-container {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }}
    
    /* ==================== FOOTER ==================== */
    .footer-container {{
        background-color: #f8f9fa;
        padding: 30px 20px;
        margin-top: 40px;
        border-top: 2px solid {COLORS['primary']};
    }}
    
    .footer-content {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    .footer-row {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 30px;
        margin-bottom: 30px;
    }}
    
    .footer-section h3 {{
        color: {COLORS['primary']};
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 12px;
        border-bottom: 2px solid {COLORS['accent']};
        padding-bottom: 8px;
    }}
    
    .footer-links {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    
    .footer-links li {{
        margin-bottom: 8px;
    }}
    
    .footer-links a {{
        color: {COLORS['primary']};
        text-decoration: none;
        font-size: 14px;
        transition: color 0.3s;
    }}
    
    .footer-links a:hover {{
        color: {COLORS['accent']};
        text-decoration: underline;
    }}
    
    .footer-bottom {{
        border-top: 1px solid #ddd;
        padding-top: 20px;
        text-align: center;
        color: #666;
        font-size: 12px;
    }}
    
    .platform-badge {{
        display: inline-block;
        background-color: {COLORS['accent']};
        color: {COLORS['primary']};
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: bold;
        margin-left: 5px;
    }}
    
    /* ==================== RESPONSIVE DESIGN ==================== */
    @media (max-width: 768px) {{
        .hero-title {{
            font-size: 1.8rem;
        }}
        
        .hero-subtitle {{
            font-size: 1rem;
        }}
        
        .footer-row {{
            grid-template-columns: 1fr;
        }}
        
        .metric-card {{
            padding: 1rem;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
    }}
    
    @media (max-width: 480px) {{
        .hero-title {{
            font-size: 1.5rem;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
        }}
        
        .hero-header {{
            padding: 1rem;
        }}
    }}
    
    /* ==================== ANIMATIONS ==================== */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.5s ease-in;
    }}
    
    /* ==================== SCROLLBAR STYLING ==================== */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #f1f1f1;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['secondary']};
    }}
    </style>
    """

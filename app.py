"""
Employee Attrition Risk Prediction System - F500 Corporate Dashboard
======================================================================
Enterprise-grade dashboard designed for top-tier financial and consulting firms.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime
from predict import (
    predict_attrition_risk,
    get_employee_by_id,
    get_high_risk_employees,
    get_risk_summary,
    get_department_summary,
    parse_daily_log_for_chart,
    parse_history_for_chart
)
from llm_chatbot import LLMHRChatbot

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="HR Risk Analysis | Enterprise Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Employee Attrition Risk Prediction System v2.0"
    }
)

# =============================================================================
# F500 CORPORATE STYLING - Clean Corporate Data Aesthetic
# =============================================================================
st.markdown("""
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    body {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 2rem !important; 
        background-color: #0E1117 !important;
        max-width: 100% !important;
    }
    
    .main {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    
    /* Force dark background everywhere */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > div,
    [data-testid="block-container"],
    [data-testid="stVerticalBlock"] {
        background-color: #0E1117 !important;
    }
    
    /* Hide Streamlit branding but keep header visible for buttons */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Logo on right side of navigation ribbon - same size as buttons */
    section[data-testid="stTabs"] {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        flex: 1 !important;
    }
    
    .top-logo-container {
        position: absolute !important;
        top: 50% !important;
        right: 20px !important;
        transform: translateY(-50%) !important;
        z-index: 1000 !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
        display: flex !important;
        align-items: center !important;
        height: auto !important;
    }
    
    .top-logo-container img {
        height: 50px !important;
        width: auto !important;
        max-height: 50px !important;
        min-height: 50px !important;
        object-fit: contain !important;
        display: block !important;
    }
    
    /* Fix top ribbon visibility: ensure tabs/header are above content and not overlaid */
    section[data-testid="stTabs"],
    header,
    [data-testid="stHeader"],
    .main-header {
        background: transparent !important;
        z-index: 2000 !important;
        position: relative !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        z-index: 2001 !important;
    }
    
    /* Disable any pseudo-element overlays that may hide the ribbon */
    section[data-testid="stTabs"]::before,
    section[data-testid="stTabs"]::after,
    .top-logo-container::before,
    .top-logo-container::after {
        display: none !important;
        content: none !important;
    }
    
    /* Reduce top gap above the ribbon and remove extra black band */
    .block-container {
        /* keep a small padding so content doesn't touch the header but avoid large gap */
        padding-top: 0.4rem !important;
        margin-top: 0 !important;
    }
    /* Pull the tabs header upward slightly to remove residual spacing */
    section[data-testid="stTabs"] {
        margin-top: -0.6rem !important;
        padding-top: 0 !important;
        z-index: 2001 !important;
        position: relative !important;
    }
    header, [data-testid="stHeader"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Headers - WHITE TEXT */
    h1, h2, h3, h4, h5, h6, .main-header {
        color: #FFFFFF !important;
    }
    
    /* Force ALL text to be WHITE */
    p, span, div, label, li, td, th, strong, em, b, i {
        color: #FFFFFF !important;
    }
    
    /* Cards - DARK BACKGROUND */
    .metric-card {
        background: #1E293B !important;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        color: #FFFFFF !important;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60A5FA !important;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 400;
        color: #CCCCCC !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Input Fields - DARK BACKGROUND */
    .stSelectbox > div > div, .stTextInput input, .stNumberInput input {
        background-color: #1E293B !important;
        border: 1px solid #475569 !important;
        color: #FFFFFF !important;
    }
    
    .stMultiSelect > div > div > div {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
    }
    
    /* Tables */
    .dataframe {
        border: 1px solid #334155;
        border-radius: 6px;
        overflow: hidden;
    }
    .dataframe thead th {
        background-color: #1E293B;
        color: #FFFFFF;
        padding: 1rem;
    }
    .dataframe tbody td {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
        border-bottom: 1px solid #334155;
    }
    
    /* Data Field Box - Enhanced */
    .data-box {
        background: linear-gradient(135deg, #1E293B 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    
    .data-box:hover {
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transform: translateY(-1px);
    }
    
    .data-label {
        color: #94A3B8 !important;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .data-value {
        color: #FFFFFF !important;
        font-size: 1.15rem;
        font-weight: 600;
        line-height: 1.4;
    }
    
    /* Qualitative Feedback Box */
    .qualitative-box {
        background: linear-gradient(135deg, #1E293B 0%, #1E293B 100%);
        border-left: 4px solid #007AFF;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .qualitative-label {
        font-size: 0.75rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .qualitative-content {
        font-size: 0.95rem;
        color: #E5E5E5 !important;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    
    /* =========================================================================
       GEMINI CHATBOT SPECIFIC STYLING
       ========================================================================= */
       
    /* 1. MESSAGES CONTAINER */
    .chat-messages-container {
        /* Height calc maximized */
        height: 70vh; 
        overflow-y: auto !important;
        /* Padding to clear the input bar pill */
        padding: 10px 20px 120px 20px !important; 
        scroll-behavior: smooth;
        margin-top: 0px;
        background-color: #0E1117;
        position: relative;
        z-index: 1;
        display: block !important;
    }
    
    .chat-messages-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-messages-container::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }

    /* Messages */
    .message {
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        width: 100%;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message-user {
        align-items: flex-end;
    }
    
    .message-bubble-user {
        background-color: #282A2C;
        color: #E3E3E3 !important;
        border-radius: 24px;
        padding: 12px 24px;
        max-width: 80%;
        text-align: left;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .message-assistant {
        align-items: flex-start;
    }
    
    .message-bubble-assistant {
        background: transparent;
        color: #E3E3E3 !important;
        padding: 0;
        max-width: 90%;
        text-align: left;
        font-size: 15px;
        line-height: 1.6;
    }
    
    .assistant-icon {
        font-size: 24px;
        margin-right: 16px;
        margin-top: -2px;
        background: linear-gradient(135deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        vertical-align: top;
    }
    
    /* 2. FLOATING INPUT BAR - FIXED TO BOTTOM - TRANSPARENT CONTAINER */
    .st-key-chat_input_form {
        position: fixed !important;
        bottom: 20px !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 100000 !important; 
        background: transparent !important; 
        padding: 0px !important;
        display: flex !important;
        justify-content: center !important;
        pointer-events: none; 
    }
    
    /* The pill container itself */
    .st-key-chat_input_form > div {
        width: 70% !important;
        max-width: 800px !important;
        pointer-events: auto; 
        background-color: #1E1F20 !important;
        border: 1px solid #444746 !important;
        border-radius: 36px !important;
        padding: 6px 16px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        transition: border-color 0.2s;
    }
    
    .st-key-chat_input_form > div:focus-within {
        border-color: #A8C7FA !important;
        background-color: #1E1F20 !important;
    }
    
    /* Input Styling */
    .st-key-chat_input_form input {
        background: transparent !important;
        border: none !important;
        color: #E3E3E3 !important;
        font-size: 16px !important;
        padding: 8px 0 !important;
    }
    
    /* Submit Button Styling */
    .st-key-chat_input_form button {
        background: transparent !important;
        border: none !important;
        color: #A8C7FA !important;
        padding: 0 10px !important;
        margin-top: 2px !important;
    }
    .st-key-chat_input_form button:hover {
        background: rgba(168, 199, 250, 0.1) !important;
        color: #D3E3FD !important;
    }
    
    /* Loading Animation */
    .loading-dots {
        display: inline-flex;
        gap: 6px;
        margin-left: 44px; /* Align with text */
        margin-bottom: 20px;
    }
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #A8C7FA;
        animation: loadingPulse 1.4s ease-in-out infinite;
    }
    .loading-dot:nth-child(2) { animation-delay: 0.2s; }
    .loading-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes loadingPulse {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1); }
    }
    
    /* Hide the form outline/default streamlit style if leaked */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    
    /* Tab Styling - DARK THEME */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #1E293B !important;
        padding: 0.6rem 0.75rem;
        border-radius: 10px;
        border: 1px solid #334155;
        box-shadow: 0 8px 18px rgba(0,0,0,0.3);
        align-items: center;
        justify-content: flex-start;
        position: sticky;
        top: 0;
        z-index: 100;
        margin-bottom: 0.5rem !important; /* Reduced bottom margin */
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
        color: #CCCCCC !important;
        background: #334155 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #475569 !important;
        color: #FFFFFF !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #60A5FA !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 12px rgba(96,165,250,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
PRIMARY_BLUE = '#007ACC'
RISK_COLORS = {
    'Low': '#28A745',
    'Moderate': '#FFC107',
    'High': '#FF9800',
    'Critical': '#DC3545'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def safe_parse_json(json_str):
    """Safely parse JSON string, return empty list/dict on error."""
    if pd.isna(json_str) or not json_str or json_str == '':
        return []
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except:
        return []

def load_data():
    """Load the employee dataset with error handling."""
    try:
        df = pd.read_csv('employee_attrition_dataset_final.csv')
        return df
    except:
        return pd.DataFrame() # Fallback

@st.cache_data
def prepare_dashboard_data(df):
    """
    Process dataframe to include risk predictions for visualization.
    Cached for performance to avoid re-running rules/models on every refresh.
    """
    if df.empty: return pd.DataFrame()
    
    processed_rows = []
    # Limit processing for speed if dataset is huge, otherwise process all
    # HR datasets are usually small enough (<50k) to process fully or sample
    target_df = df if len(df) < 5000 else df.sample(5000)
    
    for idx, row in target_df.iterrows():
        try:
            # We assume predict_attrition_risk returns a dictionary with key fields
            pred = predict_attrition_risk(row)
            if pred:
                processed_rows.append({
                    'Department': pred.get('department', 'Unknown'),
                    'Risk_Level': pred.get('risk_level', 'Low'),
                    'Risk_Score': pred.get('risk_score', 0),
                    'Tenure': row.get('Tenure_Years', 0),
                    'Job_Level': row.get('Level', 'Unknown')
                })
        except:
            continue
            
    return pd.DataFrame(processed_rows)

def safe_get_driver(prediction, index=0):
    if not prediction: return None
    drivers = prediction.get('top_3_drivers', [])
    if isinstance(drivers, list) and len(drivers) > index:
        return drivers[index]
    return None

def format_risk_badge(risk_level):
    badge_class = f"risk-badge-{risk_level.lower()}"
    return f'<span class="risk-badge {badge_class}">{risk_level}</span>'

# =============================================================================
# DATA LOADING
# =============================================================================
if 'df' not in st.session_state:
    st.session_state.df = load_data()

if 'chatbot' not in st.session_state:
    try:
        llm_chatbot = LLMHRChatbot(st.session_state.df)
        st.session_state.chatbot = llm_chatbot
        if not llm_chatbot.client:
            st.session_state.chatbot_available = False
        else:
            st.session_state.chatbot_available = True
    except Exception as e:
        st.session_state.chatbot = None
        st.session_state.chatbot_available = False

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

df = st.session_state.df

# =============================================================================
# TOP NAVIGATION TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🔍 Search & Profile",
    "📈 Department Analysis",
    "💬 AI Assistant"
])

# =============================================================================
# LOGO - Right Side of Navigation Ribbon
# =============================================================================
try:
    import base64
    from pathlib import Path
    import os
    
    # Try multiple possible paths for the logo
    possible_paths = [
        Path("logo.png"),
        Path("AI Class Project/logo.png"),
        Path(os.path.join(os.path.dirname(__file__), "logo.png")) if '__file__' in globals() else None
    ]
    
    logo_path = None
    for path in possible_paths:
        if path and path.exists():
            logo_path = path
            break
    
    if logo_path and logo_path.exists():
        # Read and encode logo
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        
        st.markdown(f"""
            <div class="top-logo-container" id="nav-logo">
                <img src="data:image/png;base64,{logo_data}" alt="Logo">
            </div>
            <script>
                (function() {{
                    function positionLogo() {{
                        var tabsContainer = document.querySelector('section[data-testid="stTabs"]');
                        var tabList = tabsContainer ? tabsContainer.querySelector('[data-baseweb="tab-list"]') : null;
                        var logo = document.getElementById('nav-logo');
                        
                        if (tabsContainer && tabList && logo) {{
                            // Position logo relative to tab list container
                            var tabListRect = tabList.getBoundingClientRect();
                            var tabsContainerRect = tabsContainer.getBoundingClientRect();
                            
                            logo.style.position = 'absolute';
                            logo.style.top = '50%';
                            logo.style.right = '15px';
                            logo.style.transform = 'translateY(-50%)';
                            logo.style.zIndex = '1001';
                            logo.style.height = '50px';
                        }}
                    }}
                    
                    // Run on load and after a delay
                    setTimeout(positionLogo, 100);
                    setTimeout(positionLogo, 500);
                    window.addEventListener('load', positionLogo);
                }})();
            </script>
        """, unsafe_allow_html=True)
except Exception as e:
    # If logo can't be loaded, continue without it
    pass

# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================
with tab1:
    st.markdown('<div class="main-header">Executive Dashboard</div>', unsafe_allow_html=True)
    if not df.empty:
        # Pre-process data for charts
        analysis_df = prepare_dashboard_data(df)
        summary = get_risk_summary(df)
        
        # Calculate estimated financial risk (Placeholder logic: 30k per critical, 15k per high risk)
        estimated_financial_risk = (summary["critical_count"] * 40000) + (summary["high_risk_count"] * 20000)
        formatted_risk_cost = f"${estimated_financial_risk:,.0f}"

        # --- KPI Cards ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{summary["total_employees"]:,}</div><div class="metric-label">Total Employees</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#FF9800">{summary["high_risk_count"]}</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#DC3545">{summary["critical_count"]}</div><div class="metric-label">Critical</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#A8C7FA">{formatted_risk_cost}</div><div class="metric-label">Est. Attrition Cost</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # --- ROW 1: General Overview (Existing 2) ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 1. Global Risk Distribution")
            rd = summary['risk_distribution']
            fig1 = px.pie(names=list(rd.keys()), values=list(rd.values()), hole=0.6, color_discrete_sequence=[RISK_COLORS.get(k, '#333') for k in rd.keys()])
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFF', height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.markdown("### 2. Average Risk by Department")
            ds = get_department_summary(df)
            ddf = pd.DataFrame([{'Dept': k, 'Score': v['avg_score']} for k,v in ds.items()])
            fig2 = px.bar(ddf, x='Dept', y='Score', color_discrete_sequence=[PRIMARY_BLUE], text_auto='.1f')
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFF', height=320, margin=dict(l=20, r=20, t=20, b=20))
            fig2.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig2, use_container_width=True)

        # --- ROW 2: Deep Dive (New 2) ---
        c3, c4 = st.columns(2)
        
        with c3:
            st.markdown("### 3. Risk Composition by Department")
            # NEW CHART: Stacked Bar showing counts of Low/High/Critical per Dept
            if not analysis_df.empty:
                risk_comp = analysis_df.groupby(['Department', 'Risk_Level']).size().reset_index(name='Count')
                # Sort Risk Level for consistent ordering
                risk_order = ['Low', 'Moderate', 'High', 'Critical']
                fig3 = px.bar(risk_comp, x='Department', y='Count', color='Risk_Level', 
                              color_discrete_map=RISK_COLORS, category_orders={'Risk_Level': risk_order})
                fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFF', height=320, margin=dict(l=20, r=20, t=20, b=20), barmode='stack')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Insufficient data for composition chart.")

        with c4:
            st.markdown("### 4. Attrition Risk Curve (Tenure)")
            # NEW CHART: Line chart of Avg Risk vs Tenure
            if not analysis_df.empty:
                # Bin tenure to avoid noise if needed, or just round
                analysis_df['Tenure_Round'] = analysis_df['Tenure'].apply(lambda x: round(x))
                tenure_trend = analysis_df.groupby('Tenure_Round')['Risk_Score'].mean().reset_index()
                tenure_trend = tenure_trend[tenure_trend['Tenure_Round'] < 15] # Filter outliers > 15 years for cleaner chart
                
                fig4 = px.line(tenure_trend, x='Tenure_Round', y='Risk_Score', markers=True)
                fig4.update_traces(line_color='#FF9800', line_width=3)
                fig4.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFF', height=320, margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Tenure (Years)", yaxis_title="Avg Risk Score"
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Insufficient data for tenure analysis.")

        # --- ROW 3: Seniority & Action (New 1 + Existing Table) ---
        c5, c6 = st.columns([1, 1])
        
        with c5:
            st.markdown("### 5. Risk by Job Level")
            # NEW CHART: Horizontal Bar for Job Level
            if not analysis_df.empty:
                level_risk = analysis_df.groupby('Job_Level')['Risk_Score'].mean().reset_index()
                fig5 = px.bar(level_risk, y='Job_Level', x='Risk_Score', orientation='h', color='Risk_Score', color_continuous_scale='Bluered')
                fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFF', height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Insufficient data for job level analysis.")

        with c6:
            st.markdown("### 6. Top Retention Priorities")
            # EXISTING TABLE: Just integrated into the grid
            try:
                high_risk_list = get_high_risk_employees(df, min_level="high")
                if high_risk_list:
                    alert_df = pd.DataFrame(high_risk_list).head(5)
                    if not alert_df.empty:
                        display_cols = {'employee_name': 'Name', 'department': 'Dept', 'risk_score': 'Score', 'risk_level': 'Level'}
                        alert_df = alert_df.rename(columns=display_cols)[display_cols.values()]
                        st.dataframe(
                            alert_df,
                            use_container_width=True,
                            hide_index=True,
                            height=300,
                            column_config={"Score": st.column_config.ProgressColumn("Score", format="%.0f%%", min_value=0, max_value=100)}
                        )
                else:
                    st.info("No critical/high risk employees found.")
            except:
                st.info("Could not load alert list.")

# =============================================================================
# TAB 2: SEARCH & FILTER
# =============================================================================
with tab2:
    if 'viewing_profile' not in st.session_state: st.session_state.viewing_profile = False
    
    if st.session_state.viewing_profile and st.session_state.get('selected_employee_id'):
        if st.button("← Back to Search"):
            st.session_state.viewing_profile = False
            st.rerun()
        
        eid = st.session_state.selected_employee_id
        emp_match = df[df['Employee_ID'] == eid]
        
        if not emp_match.empty:
            emp_row = emp_match.iloc[0]
            pred = predict_attrition_risk(emp_row)
            
            # ENHANCED PROFILE HEADER WITH RISK BADGE
            risk_level = pred.get('risk_level', 'Unknown') if pred else 'Unknown'
            risk_score = pred.get('risk_score', 0) if pred else 0
            risk_colors_map = {
                'Low': '#34C759',
                'Moderate': '#FFCC00',
                'High': '#FF9500',
                'Critical': '#FF3B30'
            }
            risk_bg_colors_map = {
                'Low': 'rgba(52, 199, 89, 0.1)',
                'Moderate': 'rgba(255, 204, 0, 0.1)',
                'High': 'rgba(255, 149, 0, 0.1)',
                'Critical': 'rgba(255, 59, 48, 0.1)'
            }
            r_color = risk_colors_map.get(risk_level, '#8E8E93')
            r_bg_color = risk_bg_colors_map.get(risk_level, 'rgba(142, 142, 147, 0.1)')
            
            # Header with enhanced risk badge
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                # Display name on top, ID as small text - BIG AND READABLE
                employee_name = emp_row.get('Employee_Name', '')
                employee_id = emp_row.get('Employee_ID', eid)
                
                if employee_name:
                    st.markdown(f"<h1 style='font-size: 2.5rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem; line-height: 1.2;'>{employee_name}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size: 0.85em; color: #8E8E93; font-weight: 400;'>ID: {employee_id}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='font-size: 2.5rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem; line-height: 1.2;'>Employee {employee_id}</h1>", unsafe_allow_html=True)
                
                manager_name = emp_row.get('Manager_Name', '')
                dept = emp_row.get('Department', '')
                role = emp_row.get('JobRole', '')
                
                # Helper to check if value has data (defined later but needed here)
                def has_data_check(value, allow_zero=False):
                    if pd.isna(value):
                        return False
                    if value == '' or value == 'N/A' or value == '-':
                        return False
                    if isinstance(value, (int, float)) and value == 0 and not allow_zero:
                        return False
                    return True
                
                # Build info line without N/A
                info_parts = []
                if has_data_check(dept):
                    info_parts.append(f"**{dept}**")
                if has_data_check(role):
                    info_parts.append(f"**{role}**")
                if has_data_check(manager_name):
                    info_parts.append(f"Manager: **{manager_name}**")
                
                if info_parts:
                    st.markdown(" &nbsp; • &nbsp; ".join(info_parts))
            
            with header_col2:
                if pred:
                    st.markdown(f"""
                        <div style='background: {r_bg_color}; border: 2px solid {r_color}; border-radius: 12px; padding: 16px; text-align: center; margin-top: 10px;'>
                            <div style='font-size: 11px; color: #8E8E93; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;'>Risk Level</div>
                            <div style='font-size: 32px; font-weight: 700; color: {r_color}; margin-bottom: 4px;'>{risk_score:.0f}%</div>
                            <div style='font-size: 14px; color: {r_color}; font-weight: 600;'>{risk_level}</div>
                            <div style='margin-top: 8px;'>
                                <div style='background: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; overflow: hidden;'>
                                    <div style='background: {r_color}; height: 100%; width: {risk_score}%; transition: width 0.3s;'></div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- ADDITIONAL INFORMATION (AT TOP, EXCLUDING JSON) ---
            # Helper to check if value has data (needed for additional info check)
            def has_data_check_early(value, allow_zero=False):
                if pd.isna(value):
                    return False
                if value == '' or value == 'N/A' or value == '-':
                    return False
                if isinstance(value, (int, float)) and value == 0 and not allow_zero:
                    return False
                return True
            
            # Check for any additional fields not displayed (excluding JSON fields)
            displayed_fields = {
                'Employee_Name', 'Employee_ID', 'Department', 'JobRole', 'Manager_Name',
                'Age', 'Gender', 'Education', 'EducationLevel', 'EducationField', 'MaritalStatus',
                'JobLevel', 'OverTime', 'BusinessTravel', 'DistanceFromHome', 'NumCompaniesWorked',
                'MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel', 'PerformanceRating',
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager', 'TrainingTimesLastYear',
                'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction',
                'Daily_Log_JSON', 'Performance_History_JSON', 'Engagement_History_JSON',
                'Manager_Notes', 'Survey_Comments', 'Manager_1on1_History_JSON'
            }
            
            # Exclude all JSON fields from additional info
            json_field_patterns = ['_JSON', '_json', 'JSON', 'json']
            
            all_fields = set(emp_row.index.tolist())
            missing_fields = all_fields - displayed_fields
            
            # Filter out JSON fields
            missing_fields = {f for f in missing_fields if not any(pattern in f for pattern in json_field_patterns)}
            
            # Display any missing fields that have data
            if missing_fields:
                missing_with_data = []
                for field in missing_fields:
                    value = emp_row.get(field)
                    if has_data_check_early(value, allow_zero=True):
                        missing_with_data.append((field, value))
                
                if missing_with_data:
                    # Helper to create a data box (needed here)
                    def info_card_early(label, value):
                        return f"""
                        <div class="data-box">
                            <div class="data-label">{label}</div>
                            <div class="data-value">{value}</div>
                        </div>
                        """
                    
                    st.markdown("#### 📌 Additional Information")
                    cols = st.columns(min(len(missing_with_data), 3))
                    for idx, (field, value) in enumerate(missing_with_data):
                        with cols[idx % len(cols)]:
                            display_value = str(value)
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                if value >= 1000:
                                    display_value = f"{value:,.0f}"
                            st.markdown(info_card_early(field.replace('_', ' ').title(), display_value), unsafe_allow_html=True)
                    st.markdown("---")
            
            # --- DETAILED DATA GRID ---
            
            # Helper to check if value has data
            def has_data(value, allow_zero=False):
                if pd.isna(value):
                    return False
                if value == '' or value == 'N/A' or value == '-':
                    return False
                if isinstance(value, (int, float)) and value == 0 and not allow_zero:
                    return False
                return True
            
            # Helper to format value
            def format_value(value, field_type='text'):
                if pd.isna(value) or value == '' or value == 'N/A' or value == '-':
                    return None
                if isinstance(value, (int, float)) and value == 0:
                    return None
                if field_type == 'currency':
                    return f"${value:,.0f}"
                if field_type == 'percent':
                    return f"{value}%"
                if field_type == 'distance':
                    return f"{value} km"
                return str(value)
            
            # Helper to create a data box
            def info_card(label, value):
                return f"""
                <div class="data-box">
                    <div class="data-label">{label}</div>
                    <div class="data-value">{value}</div>
                </div>
                """
            
            # 1. Personal & Education
            personal_fields = []
            if has_data(emp_row.get('Age')): personal_fields.append(("Age", str(int(emp_row.get('Age')))))
            if has_data(emp_row.get('Gender')): personal_fields.append(("Gender", str(emp_row.get('Gender'))))
            edu_level = emp_row.get('Education') or emp_row.get('EducationLevel')
            if has_data(edu_level): personal_fields.append(("Education Level", str(edu_level)))
            if has_data(emp_row.get('EducationField')): personal_fields.append(("Education Field", str(emp_row.get('EducationField'))))
            if has_data(emp_row.get('MaritalStatus')): personal_fields.append(("Marital Status", str(emp_row.get('MaritalStatus'))))
            
            if personal_fields:
                st.markdown("#### 👤 Personal & Education")
                cols = st.columns(min(len(personal_fields), 5))
                for idx, (label, value) in enumerate(personal_fields):
                    with cols[idx % len(cols)]:
                        st.markdown(info_card(label, value), unsafe_allow_html=True)
            
            # 2. Employment
            employment_fields = []
            if has_data(emp_row.get('JobLevel')): employment_fields.append(("Job Level", str(emp_row.get('JobLevel'))))
            if has_data(emp_row.get('OverTime')): employment_fields.append(("Over Time", str(emp_row.get('OverTime'))))
            if has_data(emp_row.get('BusinessTravel')): employment_fields.append(("Business Travel", str(emp_row.get('BusinessTravel'))))
            if has_data(emp_row.get('DistanceFromHome')): employment_fields.append(("Distance", format_value(emp_row.get('DistanceFromHome'), 'distance')))
            if has_data(emp_row.get('NumCompaniesWorked')): employment_fields.append(("Num Companies", str(int(emp_row.get('NumCompaniesWorked')))))
            
            if employment_fields:
                st.markdown("#### 💼 Employment Details")
                cols = st.columns(min(len(employment_fields), 5))
                for idx, (label, value) in enumerate(employment_fields):
                    with cols[idx % len(cols)]:
                        st.markdown(info_card(label, value), unsafe_allow_html=True)

            # 3. Compensation (allow zero for stock options as it's meaningful)
            compensation_fields = []
            if has_data(emp_row.get('MonthlyIncome')): compensation_fields.append(("Monthly Income", format_value(emp_row.get('MonthlyIncome'), 'currency')))
            if has_data(emp_row.get('PercentSalaryHike')): compensation_fields.append(("Salary Hike", format_value(emp_row.get('PercentSalaryHike'), 'percent')))
            if has_data(emp_row.get('StockOptionLevel'), allow_zero=True): compensation_fields.append(("Stock Options", str(int(emp_row.get('StockOptionLevel')))))
            if has_data(emp_row.get('PerformanceRating')): compensation_fields.append(("Performance", str(emp_row.get('PerformanceRating'))))
            
            if compensation_fields:
                st.markdown("#### 💰 Compensation")
                cols = st.columns(min(len(compensation_fields), 4))
                for idx, (label, value) in enumerate(compensation_fields):
                    with cols[idx % len(cols)]:
                        st.markdown(info_card(label, value), unsafe_allow_html=True)

            # 4. History (allow zero values as they are meaningful)
            history_fields = []
            if has_data(emp_row.get('TotalWorkingYears'), allow_zero=True): history_fields.append(("Total Working Years", str(int(emp_row.get('TotalWorkingYears')))))
            if has_data(emp_row.get('YearsAtCompany'), allow_zero=True): history_fields.append(("Years at Company", str(int(emp_row.get('YearsAtCompany')))))
            if has_data(emp_row.get('YearsInCurrentRole'), allow_zero=True): history_fields.append(("Years in Role", str(int(emp_row.get('YearsInCurrentRole')))))
            if has_data(emp_row.get('YearsWithCurrManager'), allow_zero=True): history_fields.append(("Years w/ Manager", str(int(emp_row.get('YearsWithCurrManager')))))
            if has_data(emp_row.get('TrainingTimesLastYear'), allow_zero=True): history_fields.append(("Training Times", str(int(emp_row.get('TrainingTimesLastYear')))))
            
            if history_fields:
                st.markdown("#### ⏳ History & Tenure")
                cols = st.columns(min(len(history_fields), 5))
                for idx, (label, value) in enumerate(history_fields):
                    with cols[idx % len(cols)]:
                        st.markdown(info_card(label, value), unsafe_allow_html=True)

            # 5. Satisfaction
            satisfaction_fields = []
            if has_data(emp_row.get('EnvironmentSatisfaction')): satisfaction_fields.append(("Environment", str(int(emp_row.get('EnvironmentSatisfaction')))))
            if has_data(emp_row.get('JobSatisfaction')): satisfaction_fields.append(("Job Satisfaction", str(int(emp_row.get('JobSatisfaction')))))
            if has_data(emp_row.get('WorkLifeBalance')): satisfaction_fields.append(("Work Life Balance", str(int(emp_row.get('WorkLifeBalance')))))
            if has_data(emp_row.get('RelationshipSatisfaction')): satisfaction_fields.append(("Relationship", str(int(emp_row.get('RelationshipSatisfaction')))))
            
            if satisfaction_fields:
                st.markdown("#### 😊 Satisfaction (1-4)")
                cols = st.columns(min(len(satisfaction_fields), 4))
                for idx, (label, value) in enumerate(satisfaction_fields):
                    with cols[idx % len(cols)]:
                        st.markdown(info_card(label, value), unsafe_allow_html=True)

            st.markdown("---")
            
            # --- QUALITATIVE FEEDBACK SECTION ---
            qualitative_fields = []
            manager_notes = emp_row.get('Manager_Notes', '')
            survey_comments = emp_row.get('Survey_Comments', '')
            
            if has_data(manager_notes):
                qualitative_fields.append(("Manager Notes", manager_notes))
            if has_data(survey_comments):
                qualitative_fields.append(("Survey Comments", survey_comments))
            
            if qualitative_fields:
                st.markdown("#### 💬 Qualitative Feedback")
                for label, content in qualitative_fields:
                    st.markdown(f"""
                        <div class="qualitative-box">
                            <div class="qualitative-label">{label}</div>
                            <div class="qualitative-content">{content}</div>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")
            
            # --- JSON DATA VISUALIZATIONS ---
            st.markdown("#### 📊 Historical Data & Trends")
            
            # Daily Log Visualization
            daily_log_json = emp_row.get('Daily_Log_JSON', '[]')
            daily_log = safe_parse_json(daily_log_json)
            if daily_log and len(daily_log) > 0:
                try:
                    daily_data = parse_daily_log_for_chart(daily_log_json)
                    if daily_data:
                        daily_df = pd.DataFrame(daily_data)
                        fig_daily = px.line(daily_df, x='day', y='hours', 
                                           title='Daily Work Hours Trend (Last 30 Days)',
                                           labels={'day': 'Day', 'hours': 'Hours Worked'})
                        fig_daily.update_traces(line_color='#007AFF', line_width=2, marker_color='#007AFF')
                        fig_daily.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font_color='#FFF',
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_daily, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not visualize daily log: {str(e)}")
            
            # Performance History Visualization
            perf_history_json = emp_row.get('Performance_History_JSON', '[]')
            perf_history = safe_parse_json(perf_history_json)
            if perf_history and len(perf_history) > 0:
                try:
                    perf_data = parse_history_for_chart(perf_history_json, 'performance')
                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        fig_perf = px.line(perf_df, x='quarter', y='performance',
                                          title='Performance History (Quarterly)',
                                          labels={'quarter': 'Quarter', 'performance': 'Performance Score'})
                        fig_perf.update_traces(line_color='#34C759', line_width=2, marker_color='#34C759')
                        fig_perf.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font_color='#FFF',
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not visualize performance history: {str(e)}")
            
            # Engagement History Visualization
            eng_history_json = emp_row.get('Engagement_History_JSON', '[]')
            eng_history = safe_parse_json(eng_history_json)
            if eng_history and len(eng_history) > 0:
                try:
                    eng_data = parse_history_for_chart(eng_history_json, 'engagement')
                    if eng_data:
                        eng_df = pd.DataFrame(eng_data)
                        fig_eng = px.line(eng_df, x='quarter', y='engagement',
                                        title='Engagement History (Quarterly)',
                                        labels={'quarter': 'Quarter', 'engagement': 'Engagement Score'})
                        fig_eng.update_traces(line_color='#FF9500', line_width=2, marker_color='#FF9500')
                        fig_eng.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font_color='#FFF',
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_eng, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not visualize engagement history: {str(e)}")
            
            st.markdown("---")
            
            # Risk Drivers Section - Enhanced
            if pred and pred.get('top_3_drivers'):
                st.markdown("#### ⚠️ Top Risk Drivers")
                for i, driver in enumerate(pred['top_3_drivers'], 1):
                    driver_color = risk_colors_map.get(risk_level, '#8E8E93')
                    with st.expander(f"🔴 {i}. {driver['name']} ({driver['influence_pct']}% influence)", expanded=(i == 1)):
                        st.markdown(f"**Description:** {driver.get('description', 'N/A')}")
                        st.markdown(f"**Benchmark:** {driver.get('benchmark', 'N/A')}")
                        st.progress(driver['influence_pct'] / 100)
            
            st.markdown("---")
            
            # JSON Data Expanders
            col_json1, col_json2, col_json3 = st.columns(3)
            with col_json1:
                with st.expander("📝 Daily Log (JSON)"):
                    daily_log_display = safe_parse_json(emp_row.get('Daily_Log_JSON', '[]'))
                    if daily_log_display:
                        st.json(daily_log_display)
                    else:
                        st.info("No daily log data available")
            
            with col_json2:
                with st.expander("📈 Performance History (JSON)"):
                    perf_display = safe_parse_json(emp_row.get('Performance_History_JSON', '[]'))
                    if perf_display:
                        st.json(perf_display)
                    else:
                        st.info("No performance history available")
            
            with col_json3:
                with st.expander("💚 Engagement History (JSON)"):
                    eng_display = safe_parse_json(emp_row.get('Engagement_History_JSON', '[]'))
                    if eng_display:
                        st.json(eng_display)
                    else:
                        st.info("No engagement history available")

        else:
            st.error("Employee not found.")
            
    else:
        st.markdown("### Advanced Employee Search")
        
        # Filters Section
        col_filters1, col_filters2 = st.columns(2)
        with col_filters1:
            # Safe getting of department column
            dept_col_candidates = ['Department', 'department', 'Dept']
            actual_dept_col = next((c for c in dept_col_candidates if c in df.columns), 'Department')
            
            # Get unique values if column exists, else empty
            dept_options = sorted(df[actual_dept_col].unique()) if actual_dept_col in df.columns else []
            sel_dept = st.multiselect("Filter by Department", dept_options)
            
        with col_filters2:
            sel_risk = st.multiselect("Filter by Risk Level", ["Low", "Moderate", "High", "Critical"])
            
        q = st.text_input("Search Name or ID", placeholder="Type to search (e.g. 'John' or 'EMP001')...")
        
        # -------------------------------------------------------------
        # LOCAL SEARCH LOGIC (Fixing AttributeError & KeyError)
        # -------------------------------------------------------------
        if q or sel_dept or sel_risk:
            # Start with a full mask
            mask = pd.Series([True] * len(df))
            
            # Identify columns dynamically
            name_candidates = ['Employee_Name', 'Name', 'Employee Name', 'name', 'FULL_NAME']
            id_candidates = ['Employee_ID', 'ID', 'Employee ID', 'id', 'EMPLOYEE_ID']
            
            name_col = next((c for c in name_candidates if c in df.columns), None)
            id_col = next((c for c in id_candidates if c in df.columns), None)
            
            # 1. Text Search (Local pandas filtering)
            if q:
                q_str = str(q).lower()
                name_match = pd.Series([False] * len(df))
                id_match = pd.Series([False] * len(df))
                
                if name_col:
                    name_match = df[name_col].astype(str).str.lower().str.contains(q_str, na=False)
                
                if id_col:
                    id_match = df[id_col].astype(str).str.lower().str.contains(q_str, na=False)
                
                mask = mask & (name_match | id_match)
            
            # 2. Department Filter
            if sel_dept and actual_dept_col in df.columns:
                mask = mask & df[actual_dept_col].isin(sel_dept)
                
            filtered_raw = df[mask]
            
            # 3. Calculate Risk & Apply Risk Filter
            results = []
            
            # Limit iteration for performance if no specific text query
            max_process = 200
            
            for _, row in filtered_raw.head(max_process).iterrows():
                try:
                    p = predict_attrition_risk(row)
                    if not p: continue
                    
                    # Risk Filter Check
                    if sel_risk and p['risk_level'] not in sel_risk:
                        continue
                        
                    results.append(p)
                except:
                    continue
            
            st.caption(f"Showing top {len(results)} matches")
            
            if not results:
                st.info("No employees found matching your criteria.")
            
            # Display Results
            for r in results[:20]: # Show first 20 cards
                r_color = RISK_COLORS.get(r['risk_level'], '#FFF')
                with st.expander(f"**{r['employee_name']}** •  {r['department']}"):
                    col_details, col_driver, col_action = st.columns([2, 2, 1.5])
                    
                    with col_details:
                        st.markdown(f"🆔 **{r.get('employee_id', 'N/A')}**")
                        st.markdown(f"💼 {r.get('level', 'N/A')}")
                        st.markdown(f"⏳ {r.get('tenure_years', 0):.1f} Years")
                        
                    with col_driver:
                        drivers = r.get('top_3_drivers', [])
                        if drivers:
                            d = drivers[0]
                            st.markdown(f"**Top Risk Factor:**")
                            st.caption(f"{d['name']}")
                            st.progress(d['influence_pct'] / 100)
                        else:
                            st.caption("No specific risk drivers detected")
                            
                    with col_action:
                        st.markdown(f"<div style='text-align:right; line-height:1;'><span style='font-size:24px; color:{r_color}; font-weight:bold'>{r['risk_score']:.0f}%</span><br><span style='font-size:12px; opacity:0.8'>{r['risk_level']}</span></div>", unsafe_allow_html=True)
                        st.write("")
                        if st.button("View Profile", key=f"btn_{r['employee_id']}", use_container_width=True):
                            st.session_state.selected_employee_id = r['employee_id']
                            st.session_state.viewing_profile = True
                            st.rerun()

# =============================================================================
# TAB 3: DEPARTMENTS
# =============================================================================
with tab3:
    st.markdown("### Department Analysis")
    dept = st.selectbox("Select Department", sorted(df['Department'].unique()))
    
    if dept:
        dept_employees = df[df['Department'] == dept].copy()
        
        # Calculate risk scores for display (fixing KeyError)
        dept_risk_list = []
        for idx, emp_row in dept_employees.iterrows():
            try:
                # Use the prediction function to get calculated fields
                pred = predict_attrition_risk(emp_row)
                if pred:
                    dept_risk_list.append({
                        'Employee_ID': pred.get('employee_id', 'N/A'),
                        'Employee_Name': pred.get('employee_name', 'N/A'),
                        'Risk_Score': pred.get('risk_score', 0),
                        'Risk_Level': pred.get('risk_level', 'N/A'),
                        'Tenure': emp_row.get('Tenure_Years', 0) # Grab tenure from raw df if possible
                    })
            except:
                continue
        
        # Create a DF for the charts/tables
        if dept_risk_list:
            display_df = pd.DataFrame(dept_risk_list)
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Employees", len(dept_employees))
            avg_score = display_df['Risk_Score'].mean()
            c2.metric("Avg Risk Score", f"{avg_score:.1f}%")
            high_risk_n = len(display_df[display_df['Risk_Score'] > 70])
            c3.metric("High Risk Count", high_risk_n)
            
            st.markdown("---")
            
            # New Feature: Tenure vs Risk Chart
            col_chart, col_table = st.columns([1, 1])
            
            with col_chart:
                st.markdown("#### Risk vs. Tenure")
                if 'Tenure' in display_df.columns:
                    fig = px.scatter(
                        display_df, 
                        x='Tenure', 
                        y='Risk_Score', 
                        color='Risk_Level',
                        color_discrete_map=RISK_COLORS,
                        hover_data=['Employee_Name'],
                        title="Is attrition driven by new hires or veterans?"
                    )
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFF', height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tenure data not available for plotting.")

            with col_table:
                st.markdown("#### Employee List")
                st.dataframe(
                    display_df[['Employee_Name', 'Risk_Score', 'Risk_Level']].sort_values('Risk_Score', ascending=False), 
                    use_container_width=True,
                    height=350,
                    column_config={
                        "Risk_Score": st.column_config.NumberColumn("Risk Score", format="%.1f%%")
                    }
                )
        else:
            st.info("No risk data available for this department.")

# =============================================================================
# TAB 4: AI ASSISTANT (Gemini Style)
# =============================================================================
with tab4:
    # 1. MESSAGES CONTAINER
    # The ID 'chatMessages' is important for JS scrolling
    st.markdown('<div class="chat-messages-container" id="chatMessages">', unsafe_allow_html=True)
    
    if not st.session_state.chat_messages:
        # PADDING REMOVED - WELCOME MESSAGE AT TOP
        st.markdown("""
            <div style="text-align: center; margin-top: 0px; opacity: 0.8;">
                <h1 style="background: linear-gradient(135deg, #4285F4, #9B72CB, #D96570); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600; font-size: 3rem; margin-bottom: 10px;">Hello, HR Team</h1>
                <p style="font-size: 1.2rem; color: #94A3B8;">How can I help you analyze risk today?</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show placeholder message if chatbot is unavailable
        if not st.session_state.get('chatbot_available', False):
            st.markdown("""
                <div style="margin-top: 20px; padding: 20px; background-color: #1E293B; border: 1px solid #FF9800; border-radius: 12px; text-align: left;">
                    <div style="font-size: 1.1rem; color: #FF9800; font-weight: 600; margin-bottom: 10px;">⚠️ LLM Chatbot Unavailable</div>
                    <div style="color: #CCCCCC; line-height: 1.6;">
                        <p>To use the AI Assistant, please configure your Groq API key in <code style="background-color: #0E1117; padding: 2px 6px; border-radius: 4px;">config.env</code></p>
                        <p style="margin-top: 10px;">Add the following line:</p>
                        <pre style="background-color: #0E1117; padding: 12px; border-radius: 8px; margin-top: 8px; overflow-x: auto;"><code>GROQ_API_KEY=your_api_key_here</code></pre>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    for msg in st.session_state.chat_messages:
        role = msg['role']
        content = msg['content'].replace('\n', '<br>')
        ts = msg.get('timestamp', '')
        
        if role == 'user':
            st.markdown(f"""
                <div class="message message-user">
                    <div class="message-bubble message-bubble-user">{content}</div>
                    <div class="message-time">{ts}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Check if it's a chart
            if msg.get('type') == 'chart':
                 st.markdown(f"""
                    <div class="message message-assistant">
                        <div style="display:flex">
                            <div class="assistant-icon">✨</div>
                            <div style="flex:1">
                                <div class="message-bubble message-bubble-assistant">Here is the chart you requested:</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                 st.plotly_chart(msg['chart_data'], use_container_width=True)
            else:
                st.markdown(f"""
                    <div class="message message-assistant">
                        <div style="display:flex">
                            <div class="assistant-icon">✨</div>
                            <div style="flex:1">
                                <div class="message-bubble message-bubble-assistant">{content}</div>
                                <div class="message-time" style="padding-left:0">{ts}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    if st.session_state.get('chatbot_processing', False):
        st.markdown("""
            <div class="loading-dots">
                <div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # End chat-messages-container

    # 2. FLOATING INPUT FORM
    # Using st.form with a specific key to target it with CSS
    with st.form("chat_input_form", clear_on_submit=True, border=False):
        col_in, col_btn = st.columns([1, 0.1])
        with col_in:
            user_input = st.text_input("Message", placeholder="Message AI Assistant...", label_visibility="collapsed")
        with col_btn:
            sent = st.form_submit_button("➤")
    
    if sent and user_input:
         st.session_state.chat_messages.append({'role': 'user', 'content': user_input, 'timestamp': datetime.now().strftime("%H:%M")})
         st.session_state.chatbot_processing = True
         st.rerun()

    # If processing, run the logic (AFTER rerun to show user message first)
    if st.session_state.get('chatbot_processing', False) and st.session_state.chat_messages[-1]['role'] == 'user':
        if not st.session_state.get('chatbot_available', False) or st.session_state.chatbot is None:
            # Chatbot not available - show error
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': '⚠️ **LLM Chatbot Unavailable**\n\nPlease configure your Groq API key to use the AI Assistant. See the configuration instructions above.',
                'timestamp': datetime.now().strftime("%H:%M")
            })
            st.session_state.chatbot_processing = False
            st.rerun()
        else:
            # Process query normally
            chatbot = st.session_state.chatbot
            last_query = st.session_state.chat_messages[-1]['content']
            response = chatbot.process_query(last_query, {'view_type': 'chatbot'}, st.session_state.chat_messages)
            
            # Add response
            msg_data = {
                'role': 'assistant',
                'timestamp': datetime.now().strftime("%H:%M")
            }
            
            if response.get('type') == 'chart':
                msg_data['type'] = 'chart'
                msg_data['chart_data'] = response['chart_data']
                msg_data['content'] = ''
            else:
                msg_data['content'] = response.get('content', '')
                
            st.session_state.chat_messages.append(msg_data)
            st.session_state.chatbot_processing = False
            st.rerun()

    # 3. DYNAMIC LAYOUT JAVASCRIPT - FIXED SCROLLING
    st.markdown("""
    <script>
    (function() {
        function updateLayout() {
            var tabs = document.querySelectorAll('[data-baseweb="tab"]');
            var isAiTab = false;
            
            // Check if AI Assistant tab is selected
            for (var i = 0; i < tabs.length; i++) {
                if (tabs[i].getAttribute('aria-selected') === 'true' && 
                    tabs[i].textContent.includes('AI Assistant')) {
                    isAiTab = true;
                    break;
                }
            }
            
            var mainContainer = document.querySelector('.main .block-container');
            var body = document.body;
            var html = document.documentElement;
            
            if (isAiTab) {
                // Lock scrolling for Chatbot
                if (mainContainer) {
                    mainContainer.style.height = '100vh';
                    mainContainer.style.overflow = 'hidden';
                    mainContainer.style.paddingBottom = '0px';
                }
                body.style.overflow = 'hidden';
                html.style.overflow = 'hidden';
                
                // Adjust chat container height
                var chatContainer = document.getElementById('chatMessages');
                if (chatContainer) {
                    // Force height calc
                    chatContainer.style.height = 'calc(100vh - 200px)';
                }
                
                // Ensure input bar is visible
                var inputBar = document.querySelector('.st-key-chat_input_form');
                if (inputBar) {
                    inputBar.style.display = 'flex';
                }
                
            } else {
                // Unlock scrolling for other tabs
                if (mainContainer) {
                    mainContainer.style.height = 'auto';
                    mainContainer.style.overflow = 'visible';
                    mainContainer.style.paddingBottom = '2rem';
                }
                body.style.overflow = 'auto';
                html.style.overflow = 'auto';
            }
        }
        
        // Run constantly to catch tab switches
        setInterval(updateLayout, 500);
        
        // Run on click
        document.addEventListener('click', function() {
            setTimeout(updateLayout, 100);
        });
        
        // Auto-scroll chat
        function scrollToBottom() {
            const msgContainer = document.getElementById('chatMessages');
            if(msgContainer) {
                msgContainer.scrollTop = msgContainer.scrollHeight;
            }
        }
        
        // Watch for new messages
        const msgContainer = document.getElementById('chatMessages');
        if(msgContainer) {
            const observer = new MutationObserver(() => {
                scrollToBottom();
            });
            observer.observe(msgContainer, { childList: true, subtree: true, characterData: true });
        }
    })();
    </script>
    """, unsafe_allow_html=True)
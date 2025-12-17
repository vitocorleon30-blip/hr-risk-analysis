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
from predict import (
    predict_attrition_risk,
    get_employee_by_id,
    get_high_risk_employees,
    get_risk_summary,
    get_department_summary,
    parse_daily_log_for_chart,
    parse_history_for_chart
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="HR Risk Analysis | Enterprise Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
    
    .main {
        background-color: #FFFFFF;
        padding: 2rem 2rem 2rem 1rem;
    }
    
    /* Sidebar - Dark Navigation */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        padding: 1.5rem 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #CCCCCC;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        transition: all 0.2s;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label {
        background-color: rgba(0, 122, 204, 0.15);
        color: #007ACC;
        border-left: 3px solid #007ACC;
        padding-left: calc(1rem - 3px);
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #FFFFFF;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stSelectbox select {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #CCCCCC;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headers */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #333333;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #333333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #007ACC;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 400;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk Badges - Pill-shaped */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-badge-low {
        background-color: #28A745;
        color: #FFFFFF;
    }
    
    .risk-badge-moderate {
        background-color: #FFC107;
        color: #333333;
    }
    
    .risk-badge-high {
        background-color: #FF9800;
        color: #FFFFFF;
    }
    
    .risk-badge-critical {
        background-color: #DC3545;
        color: #FFFFFF;
    }
    
    /* Progress Bars - Minimalist */
    .progress-container {
        margin: 1rem 0;
    }
    
    .progress-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 0.5rem;
    }
    
    .progress-description {
        font-size: 0.8rem;
        color: #666666;
        margin-top: 0.5rem;
        padding-left: 0.5rem;
        font-style: italic;
    }
    
    /* Input Fields */
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 4px;
    }
    
    .stSelectbox label {
        color: #333333;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Tables */
    .dataframe {
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        overflow: hidden;
    }
    
    .dataframe thead th {
        background-color: #1E293B;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 1rem;
    }
    
    .dataframe tbody td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #F0F0F0;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #F8F9FA;
        border-left: 4px solid #007ACC;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #333333;
        line-height: 1.6;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E0E0E0;
        margin: 2rem 0;
    }
    
    /* Chart Container */
    .chart-container {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# COLOR CONSTANTS - F500 Corporate Palette
# =============================================================================
PRIMARY_BLUE = '#007ACC'
DARK_NAV = '#1E293B'
TEXT_DARK = '#333333'
TEXT_LIGHT = '#666666'
BORDER_LIGHT = '#CCCCCC'
BORDER_SUBTLE = '#E0E0E0'
BG_WHITE = '#FFFFFF'

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
def load_data():
    """Load the employee dataset with error handling."""
    try:
        df = pd.read_csv('employee_attrition_dataset_final.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure 'employee_attrition_dataset_final.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

def safe_parse_json(json_str, default=None):
    """Safely parse JSON string with error handling."""
    if pd.isna(json_str) or json_str == '' or json_str is None:
        return default
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except (json.JSONDecodeError, TypeError, ValueError):
        return default

def safe_get_driver(prediction, index=0):
    """Safely get risk driver from prediction, handling empty lists."""
    if not prediction:
        return None
    drivers = prediction.get('top_3_drivers', [])
    if isinstance(drivers, list) and len(drivers) > index:
        return drivers[index]
    return None

def format_risk_badge(risk_level):
    """Format risk level as HTML badge."""
    badge_class = f"risk-badge-{risk_level.lower()}"
    return f'<span class="risk-badge {badge_class}">{risk_level}</span>'

# =============================================================================
# DATA LOADING
# =============================================================================
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# =============================================================================
# SIDEBAR NAVIGATION - Dark Corporate Style
# =============================================================================
st.sidebar.markdown("""
    <div style='text-align: left; padding: 1.5rem 1rem 2rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.1);'>
        <h1 style='color: #FFFFFF; font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;'>HR Risk Analysis</h1>
        <p style='color: #CCCCCC; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>Enterprise Dashboard</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "👤 Employee Profile", "📈 Department Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style='text-align: left; color: #CCCCCC; font-size: 0.75rem; padding: 1rem; border-top: 1px solid rgba(255,255,255,0.1);'>
        <p style='margin: 0; color: #FFFFFF; font-weight: 500;'>Attrition Risk System</p>
        <p style='margin: 0.25rem 0 0 0;'>v2.0 Enterprise Edition</p>
    </div>
""", unsafe_allow_html=True)

# =============================================================================
# DASHBOARD PAGE
# =============================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">Executive Dashboard</div>', unsafe_allow_html=True)
    
    # Get summary statistics
    try:
        summary = get_risk_summary(df)
    except Exception as e:
        st.error(f"Error loading summary statistics: {str(e)}")
        st.stop()
    
    # Key metrics - Card-based layout
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{summary['total_employees']:,}</div>
                <div class="metric-label">Total Employees</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk_pct = (summary['high_risk_count'] / summary['total_employees']) * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{summary['high_risk_count']}</div>
                <div class="metric-label">High Risk Employees</div>
                <div style="font-size: 0.75rem; color: #666666; margin-top: 0.5rem;">{high_risk_pct:.1f}% of workforce</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        critical_pct = (summary['critical_count'] / summary['total_employees']) * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #DC3545;">{summary['critical_count']}</div>
                <div class="metric-label">Critical Risk</div>
                <div style="font-size: 0.75rem; color: #666666; margin-top: 0.5rem;">{critical_pct:.1f}% of workforce</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{summary['avg_risk_score']:.1f}%</div>
                <div class="metric-label">Average Risk Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts section - Clean, minimal design
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Distribution")
        risk_dist = summary['risk_distribution']
        total_employees = summary['total_employees']
        
        # Donut chart with total in center
        fig_dist = go.Figure(data=[go.Pie(
            labels=list(risk_dist.keys()),
            values=list(risk_dist.values()),
            hole=0.6,
            marker=dict(
                colors=[RISK_COLORS.get(k, PRIMARY_BLUE) for k in risk_dist.keys()],
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color=TEXT_DARK),
            hovertemplate='<b>%{label}</b><br>Employees: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_dist.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1,
                font=dict(size=11, color=TEXT_DARK)
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color=TEXT_DARK),
            margin=dict(l=0, r=120, t=20, b=0),
            annotations=[dict(
                text=f'<b style="font-size:24px; color:{PRIMARY_BLUE};">{total_employees}</b><br><span style="font-size:12px; color:{TEXT_LIGHT};">Total</span>',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### Department Risk Analysis")
        try:
            dept_summary = get_department_summary(df)
            dept_data = []
            for dept, stats in dept_summary.items():
                dept_data.append({
                    'Department': dept,
                    'Avg Risk Score': stats['avg_score'],
                    'High Risk %': stats['high_risk_pct']
                })
            dept_df = pd.DataFrame(dept_data)
            
            # Clean bar chart - single accent color, thin bars, labels above
            fig_dept = go.Figure(data=[
                go.Bar(
                    x=dept_df['Department'],
                    y=dept_df['Avg Risk Score'],
                    marker=dict(
                        color=PRIMARY_BLUE,
                        line=dict(color='#FFFFFF', width=1)
                    ),
                    text=[f"{x:.1f}%" for x in dept_df['Avg Risk Score']],
                    textposition='outside',
                    textfont=dict(color=TEXT_DARK, size=11, weight='bold'),
                    hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}%<extra></extra>'
                )
            ])
            
            fig_dept.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color=TEXT_DARK),
                xaxis=dict(
                    title=dict(text="Department", font=dict(size=12, color=TEXT_DARK, weight=600)),
                    gridcolor='#F0F0F0',
                    showgrid=True,
                    gridwidth=1
                ),
                yaxis=dict(
                    title=dict(text="Average Risk Score (%)", font=dict(size=12, color=TEXT_DARK, weight=600)),
                    gridcolor='#F0F0F0',
                    showgrid=True,
                    gridwidth=1
                ),
                margin=dict(l=0, r=0, t=20, b=0),
                bargap=0.4
            )
            st.plotly_chart(fig_dept, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading department analysis: {str(e)}")
    
    st.markdown("---")
    
    # High-risk employees table
    st.markdown("### High-Risk Employee Alert")
    
    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "Moderate", "High", "Critical"],
        key="risk_filter"
    )
    
    try:
        min_level = risk_filter.lower() if risk_filter != "All" else "Moderate"
        high_risk = get_high_risk_employees(df, min_level=min_level)
        
        if high_risk:
            display_data = []
            for emp in high_risk:
                top_driver = safe_get_driver(emp, 0)
                driver_name = top_driver['name'] if top_driver else "N/A"
                driver_pct = top_driver['influence_pct'] if top_driver else 0
                
                display_data.append({
                    'Employee ID': emp.get('employee_id', 'N/A'),
                    'Name': emp.get('employee_name', 'N/A'),
                    'Department': emp.get('department', 'N/A'),
                    'Risk Score': f"{emp.get('risk_score', 0):.1f}%",
                    'Risk Level': emp.get('risk_level', 'N/A'),
                    'Trend': emp.get('risk_trend_icon', ''),
                    'Top Risk Driver': f"{driver_name} ({driver_pct}%)" if top_driver else "N/A"
                })
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("✅ No employees found matching the selected risk level.")
    except Exception as e:
        st.error(f"Error loading high-risk employees: {str(e)}")

# =============================================================================
# EMPLOYEE PROFILE PAGE
# =============================================================================
elif page == "👤 Employee Profile":
    st.markdown('<div class="main-header">Employee Risk Profile</div>', unsafe_allow_html=True)
    
    # Employee selector
    employee_ids = df['Employee_ID'].tolist()
    selected_id = st.selectbox(
        "Select Employee",
        employee_ids,
        key="employee_selector",
        help="Choose an employee to view their detailed risk profile"
    )
    
    if selected_id:
        try:
            prediction = get_employee_by_id(df, selected_id)
            employee_row = df[df['Employee_ID'] == selected_id]
            
            if employee_row.empty:
                st.error(f"Employee {selected_id} not found in the dataset.")
                st.stop()
            
            employee = employee_row.iloc[0]
        except Exception as e:
            st.error(f"Error loading employee data: {str(e)}")
            st.stop()
        
        if prediction:
            # Header section - Focal Risk Score
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {prediction.get('employee_name', 'N/A')}")
                st.markdown(f"**Employee ID:** {prediction.get('employee_id', 'N/A')}")
                st.markdown(f"**Department:** {prediction.get('department', 'N/A')} | **Level:** {prediction.get('level', 'N/A')}")
                st.markdown(f"**Manager:** {prediction.get('manager_name', 'N/A')} | **Tenure:** {prediction.get('tenure_years', 0):.1f} years")
            
            with col2:
                risk_level = prediction.get('risk_level', 'N/A')
                risk_score = prediction.get('risk_score', 0)
                risk_color = RISK_COLORS.get(risk_level, PRIMARY_BLUE)
                
                st.markdown("### Risk Assessment")
                # Large, bold risk score
                st.markdown(f"""
                    <div style="font-size: 3.5rem; font-weight: 700; color: {risk_color}; line-height: 1; margin-bottom: 1rem;">
                        {risk_score:.1f}%
                    </div>
                """, unsafe_allow_html=True)
                
                # Pill-shaped risk badge
                badge_html = format_risk_badge(risk_level)
                st.markdown(f"**Level:** {badge_html}", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="margin-top: 1rem; font-size: 0.9rem; color: {TEXT_LIGHT};">
                        {prediction.get('risk_trend_icon', '')} {prediction.get('risk_trend', 'stable').title()}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Top 3 Risk Drivers - Minimalist progress bars
            st.markdown("### Top Risk Drivers")
            
            drivers = prediction.get('top_3_drivers', [])
            if drivers and len(drivers) > 0:
                for i, driver in enumerate(drivers[:3], 1):
                    influence_pct = driver.get('influence_pct', 0)
                    st.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-label">{i}. {driver.get('name', 'N/A')} - {influence_pct}% influence</div>
                            <div style="background-color: #E0E0E0; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div style="background-color: {PRIMARY_BLUE}; height: 100%; width: {influence_pct}%; transition: width 0.3s;"></div>
                            </div>
                            <div class="progress-description">
                                {driver.get('description', 'No description available')}
                            </div>
                            <div class="progress-description" style="font-size: 0.75rem; color: #999999;">
                                {driver.get('benchmark', 'No benchmark available')}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    if i < min(3, len(drivers)):
                        st.markdown("---")
            else:
                st.info("No risk drivers identified for this employee.")
            
            # Charts section - Clean line charts
            st.markdown("### Performance & Engagement Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Daily Work Hours (Last 60 Days)")
                try:
                    daily_log = safe_parse_json(employee.get('Daily_Log_JSON', '[]'))
                    if daily_log and len(daily_log) > 0:
                        hours_data = parse_daily_log_for_chart(employee['Daily_Log_JSON'])
                        if hours_data and len(hours_data) > 0:
                            df_hours = pd.DataFrame(hours_data)
                            
                            fig_hours = go.Figure()
                            fig_hours.add_trace(go.Scatter(
                                x=df_hours['day'],
                                y=df_hours['hours'],
                                mode='lines+markers',
                                name='Work Hours',
                                line=dict(color=PRIMARY_BLUE, width=2.5),
                                marker=dict(size=5, color=PRIMARY_BLUE, line=dict(width=1, color='#FFFFFF'))
                            ))
                            fig_hours.update_layout(
                                height=350,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=11, color=TEXT_DARK),
                                xaxis=dict(
                                    title=dict(text="Day", font=dict(size=12, color=TEXT_DARK, weight=600)),
                                    gridcolor='#F0F0F0',
                                    showgrid=True
                                ),
                                yaxis=dict(
                                    title=dict(text="Hours Worked", font=dict(size=12, color=TEXT_DARK, weight=600)),
                                    gridcolor='#F0F0F0',
                                    showgrid=True
                                ),
                                margin=dict(l=0, r=0, t=20, b=0),
                                hovermode='x unified',
                                showlegend=False
                            )
                            st.plotly_chart(fig_hours, use_container_width=True)
                        else:
                            st.info("Work hours data not available.")
                    else:
                        st.info("Work hours data not available.")
                except Exception as e:
                    st.warning(f"Unable to load work hours data: {str(e)}")
            
            with col2:
                st.markdown("#### Performance Trend (3 Years)")
                try:
                    perf_history = safe_parse_json(employee.get('Performance_History_JSON', '[]'))
                    if perf_history and len(perf_history) > 0:
                        perf_data = parse_history_for_chart(employee['Performance_History_JSON'], 'performance')
                        if perf_data and len(perf_data) > 0:
                            df_perf = pd.DataFrame(perf_data)
                            
                            fig_perf = go.Figure()
                            fig_perf.add_trace(go.Scatter(
                                x=df_perf['quarter'],
                                y=df_perf['performance'],
                                mode='lines+markers',
                                name='Performance',
                                line=dict(color=PRIMARY_BLUE, width=2.5),
                                marker=dict(size=5, color=PRIMARY_BLUE, line=dict(width=1, color='#FFFFFF'))
                            ))
                            fig_perf.update_layout(
                                height=350,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=11, color=TEXT_DARK),
                                xaxis=dict(
                                    title=dict(text="Quarter", font=dict(size=12, color=TEXT_DARK, weight=600)),
                                    gridcolor='#F0F0F0',
                                    showgrid=True
                                ),
                                yaxis=dict(
                                    title=dict(text="Performance Score (1-100)", font=dict(size=12, color=TEXT_DARK, weight=600)),
                                    gridcolor='#F0F0F0',
                                    showgrid=True
                                ),
                                margin=dict(l=0, r=0, t=20, b=0),
                                hovermode='x unified',
                                showlegend=False
                            )
                            st.plotly_chart(fig_perf, use_container_width=True)
                        else:
                            st.info("Performance history not available.")
                    else:
                        st.info("Performance history not available.")
                except Exception as e:
                    st.warning(f"Unable to load performance data: {str(e)}")
            
            # Engagement trend
            st.markdown("#### Engagement Trend (3 Years)")
            try:
                eng_history = safe_parse_json(employee.get('Engagement_History_JSON', '[]'))
                if eng_history and len(eng_history) > 0:
                    eng_data = parse_history_for_chart(employee['Engagement_History_JSON'], 'engagement')
                    if eng_data and len(eng_data) > 0:
                        df_eng = pd.DataFrame(eng_data)
                        
                        fig_eng = go.Figure()
                        fig_eng.add_trace(go.Scatter(
                            x=df_eng['quarter'],
                            y=df_eng['engagement'],
                            mode='lines+markers',
                            name='Engagement',
                            line=dict(color=PRIMARY_BLUE, width=2.5),
                            marker=dict(size=5, color=PRIMARY_BLUE, line=dict(width=1, color='#FFFFFF'))
                        ))
                        fig_eng.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=11, color=TEXT_DARK),
                            xaxis=dict(
                                title=dict(text="Quarter", font=dict(size=12, color=TEXT_DARK, weight=600)),
                                gridcolor='#F0F0F0',
                                showgrid=True
                            ),
                            yaxis=dict(
                                title=dict(text="Engagement Score (1-10)", font=dict(size=12, color=TEXT_DARK, weight=600)),
                                gridcolor='#F0F0F0',
                                showgrid=True
                            ),
                            margin=dict(l=0, r=0, t=20, b=0),
                            hovermode='x unified',
                            showlegend=False
                        )
                        st.plotly_chart(fig_eng, use_container_width=True)
                    else:
                        st.info("Engagement history not available.")
                else:
                    st.info("Engagement history not available.")
            except Exception as e:
                st.warning(f"Unable to load engagement data: {str(e)}")
            
            # Text data section
            st.markdown("### Additional Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Manager Notes")
                manager_notes = employee.get('Manager_Notes', '')
                if pd.notna(manager_notes) and manager_notes and str(manager_notes).strip():
                    st.markdown(f'<div class="info-box">{manager_notes}</div>', unsafe_allow_html=True)
                else:
                    st.info("No manager notes available for this employee.")
            
            with col2:
                st.markdown("#### Survey Comments")
                survey_comments = employee.get('Survey_Comments', '')
                if pd.notna(survey_comments) and survey_comments and str(survey_comments).strip():
                    st.markdown(f'<div class="info-box">{survey_comments}</div>', unsafe_allow_html=True)
                else:
                    st.info("No survey comments available for this employee.")
            
            # Risk explanation
            st.markdown("### Risk Analysis Summary")
            explanation = prediction.get('explanation', 'No explanation available.')
            st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
        else:
            st.error(f"Unable to generate prediction for employee {selected_id}.")

# =============================================================================
# DEPARTMENT ANALYSIS PAGE
# =============================================================================
elif page == "📈 Department Analysis":
    st.markdown('<div class="main-header">Department Risk Analysis</div>', unsafe_allow_html=True)
    
    try:
        dept_summary = get_department_summary(df)
    except Exception as e:
        st.error(f"Error loading department summary: {str(e)}")
        st.stop()
    
    # Department selector
    departments = list(dept_summary.keys())
    if not departments:
        st.error("No departments found in the dataset.")
        st.stop()
    
    selected_dept = st.selectbox(
        "Select Department",
        departments,
        key="dept_selector",
        help="Choose a department to analyze its risk profile"
    )
    
    if selected_dept:
        try:
            dept_stats = dept_summary[selected_dept]
        except KeyError:
            st.error(f"Department {selected_dept} not found in summary data.")
            st.stop()
        
        # Department metrics - Card layout
        st.markdown("### Department Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dept_stats.get('total', 0):,}</div>
                    <div class="metric-label">Total Employees</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dept_stats.get('avg_score', 0):.1f}%</div>
                    <div class="metric-label">Average Risk Score</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dept_stats.get('high_risk_pct', 0):.1f}%</div>
                    <div class="metric-label">High Risk Percentage</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            dist = dept_stats.get('distribution', {})
            high_risk_count = dist.get('High', 0) + dist.get('Critical', 0)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #DC3545;">{high_risk_count}</div>
                    <div class="metric-label">High Risk Employees</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Risk Distribution - {selected_dept}")
            dist = dept_stats.get('distribution', {})
            total = dept_stats.get('total', 0)
            
            if dist:
                # Donut chart with total in center
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(dist.keys()),
                    values=list(dist.values()),
                    hole=0.6,
                    marker=dict(
                        colors=[RISK_COLORS.get(k, PRIMARY_BLUE) for k in dist.keys()],
                        line=dict(color='#FFFFFF', width=2)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=12, color=TEXT_DARK),
                    hovertemplate='<b>%{label}</b><br>Employees: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=11, color=TEXT_DARK),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.1,
                        font=dict(size=11, color=TEXT_DARK)
                    ),
                    margin=dict(l=0, r=120, t=20, b=0),
                    annotations=[dict(
                        text=f'<b style="font-size:24px; color:{PRIMARY_BLUE};">{total}</b><br><span style="font-size:12px; color:{TEXT_LIGHT};">Total</span>',
                        x=0.5, y=0.5,
                        font_size=16,
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No distribution data available.")
        
        with col2:
            st.markdown("### Cross-Department Comparison")
            dept_data = []
            for dept, stats in dept_summary.items():
                dept_data.append({
                    'Department': dept,
                    'Avg Risk': stats.get('avg_score', 0),
                    'High Risk %': stats.get('high_risk_pct', 0)
                })
            dept_df = pd.DataFrame(dept_data)
            
            # Clean bar chart - single accent color, thin bars, labels above
            fig_comp = go.Figure(data=[
                go.Bar(
                    x=dept_df['Department'],
                    y=dept_df['Avg Risk'],
                    marker=dict(
                        color=PRIMARY_BLUE,
                        line=dict(color='#FFFFFF', width=1)
                    ),
                    text=[f"{x:.1f}%" for x in dept_df['Avg Risk']],
                    textposition='outside',
                    textfont=dict(color=TEXT_DARK, size=11, weight='bold'),
                    hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}%<extra></extra>'
                )
            ])
            
            fig_comp.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color=TEXT_DARK),
                xaxis=dict(
                    title=dict(text="Department", font=dict(size=12, color=TEXT_DARK, weight=600)),
                    gridcolor='#F0F0F0',
                    showgrid=True
                ),
                yaxis=dict(
                    title=dict(text="Average Risk Score (%)", font=dict(size=12, color=TEXT_DARK, weight=600)),
                    gridcolor='#F0F0F0',
                    showgrid=True
                ),
                margin=dict(l=0, r=0, t=20, b=0),
                bargap=0.4
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # Employees in department table
        st.markdown(f"### Employees in {selected_dept}")
        dept_employees = df[df['Department'] == selected_dept]
        dept_risk_list = []
        
        if dept_employees.empty:
            st.info(f"No employees found in {selected_dept} department.")
        else:
            for idx, emp_row in dept_employees.iterrows():
                try:
                    pred = predict_attrition_risk(emp_row)
                    if pred:
                        top_driver = safe_get_driver(pred, 0)
                        driver_name = top_driver['name'] if top_driver else "N/A"
                        
                        dept_risk_list.append({
                            'Employee ID': pred.get('employee_id', 'N/A'),
                            'Name': pred.get('employee_name', 'N/A'),
                            'Risk Score': pred.get('risk_score', 0),
                            'Risk Level': pred.get('risk_level', 'N/A'),
                            'Top Driver': driver_name
                        })
                except Exception as e:
                    emp_id = emp_row.get('Employee_ID', f'Row {idx}')
                    st.warning(f"Error processing employee {emp_id}: {str(e)}")
                    continue
            
            if dept_risk_list:
                dept_risk_df = pd.DataFrame(dept_risk_list)
                dept_risk_df['Risk_Score_Numeric'] = dept_risk_df['Risk Score']
                dept_risk_df = dept_risk_df.sort_values('Risk_Score_Numeric', ascending=False)
                dept_risk_df = dept_risk_df.drop('Risk_Score_Numeric', axis=1)
                dept_risk_df['Risk Score'] = dept_risk_df['Risk Score'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(
                    dept_risk_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.info(f"No employee risk data available for {selected_dept} department.")

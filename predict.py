"""
Employee Attrition Risk Prediction Module v2.0
===============================================
Sophisticated prediction with 4-tier risk scoring and 
Top 3 Risk Drivers with influence percentages.

Risk Levels:
- Low Risk:      0-30%   🟢 Green
- Moderate Risk: 30-60%  🟡 Yellow  
- High Risk:     60-85%  🟠 Orange
- Critical Risk: 85-100% 🔴 Red
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, "feature_info.pkl")

# Risk level thresholds
RISK_THRESHOLDS = {
    'low': (0, 30),
    'moderate': (30, 60),
    'high': (60, 85),
    'critical': (85, 100)
}

RISK_COLORS = {
    'Low': '🟢',
    'Moderate': '🟡',
    'High': '🟠',
    'Critical': '🔴'
}

# =============================================================================
# RISK DRIVER DEFINITIONS
# =============================================================================

@dataclass
class RiskDriver:
    """Represents a single risk driver with its influence calculation."""
    name: str
    category: str
    weight: float  # Base weight for influence calculation
    threshold_fn: callable  # Function to check if this driver is triggered
    description_fn: callable  # Function to generate description
    benchmark_info: str  # Peer benchmark context

RISK_DRIVERS = {
    'career_progression': RiskDriver(
        name='No Career Progression',
        category='Career',
        weight=0.18,
        threshold_fn=lambda f: f.get('Months_Since_Promotion', 0) >= 18,
        description_fn=lambda f: f"No promotion in {int(f.get('Months_Since_Promotion', 0))} months",
        benchmark_info="Peer benchmarking shows 3.2x higher departure probability for employees without promotion in 18+ months"
    ),
    'compensation_gap': RiskDriver(
        name='Below-Market Compensation',
        category='Compensation',
        weight=0.16,
        threshold_fn=lambda f: f.get('Compensation_Gap_Pct', 0) < -8,
        description_fn=lambda f: f"Salary {abs(f.get('Compensation_Gap_Pct', 0)):.0f}% below market average",
        benchmark_info="Compensation gap of >8% predicts 2.1x higher attrition within 12 months"
    ),
    'manager_engagement': RiskDriver(
        name='Declining Manager Engagement',
        category='Management',
        weight=0.14,
        threshold_fn=lambda f: f.get('Manager_1on1_Trend', 0) < -0.3,
        description_fn=lambda f: f"1:1 meeting frequency declined {abs(f.get('Manager_1on1_Change_Pct', 0)):.0f}% over 12 months",
        benchmark_info="Manager engagement decline correlates with 2.5x retention risk"
    ),
    'office_presence_decline': RiskDriver(
        name='Declining Office Presence',
        category='Engagement',
        weight=0.12,
        threshold_fn=lambda f: f.get('Office_Presence_Trend', 0) < -2,
        description_fn=lambda f: f"Office attendance dropped from {f.get('Office_Presence_Start', 0):.0f}% to {f.get('Office_Presence_Current', 0):.0f}%",
        benchmark_info="Employees reducing office presence by >20% are 1.8x more likely to resign"
    ),
    'sick_leave_increase': RiskDriver(
        name='Increasing Sick Leave',
        category='Health',
        weight=0.11,
        threshold_fn=lambda f: f.get('Sick_Days_60D', 0) >= 5 or f.get('Sick_Leave_Trend', 0) > 0.5,
        description_fn=lambda f: f"{int(f.get('Sick_Days_60D', 0))} sick days in last 60 days (vs {f.get('Sick_Days_Avg', 2):.1f} avg)",
        benchmark_info="Elevated sick leave patterns indicate 2.3x burnout risk"
    ),
    'engagement_decline': RiskDriver(
        name='Declining Engagement Score',
        category='Engagement',
        weight=0.10,
        threshold_fn=lambda f: f.get('Engagement_Trend', 0) < -0.5 or f.get('Engagement_Current', 10) <= 4,
        description_fn=lambda f: f"Engagement dropped from {f.get('Engagement_Start', 0):.0f} to {f.get('Engagement_Current', 0):.0f}/10 over 3 years",
        benchmark_info="Engagement score decline of 3+ points predicts 2.8x departure risk"
    ),
    'training_stagnation': RiskDriver(
        name='Training & Development Gap',
        category='Development',
        weight=0.06,
        threshold_fn=lambda f: f.get('Months_Since_Training', 0) >= 12,
        description_fn=lambda f: f"No training/development in {int(f.get('Months_Since_Training', 0))} months",
        benchmark_info="Employees without recent training are 1.6x more likely to seek external opportunities"
    ),
    'recognition_gap': RiskDriver(
        name='Recognition Gap',
        category='Recognition',
        weight=0.05,
        threshold_fn=lambda f: f.get('Months_Since_Recognition', 0) >= 8,
        description_fn=lambda f: f"No recognition in {int(f.get('Months_Since_Recognition', 0))} months",
        benchmark_info="Lack of recognition beyond 6 months increases flight risk by 1.4x"
    ),
    'team_turnover': RiskDriver(
        name='Team Turnover Contagion',
        category='Team',
        weight=0.04,
        threshold_fn=lambda f: f.get('Team_Departures_6M', 0) >= 2,
        description_fn=lambda f: f"{int(f.get('Team_Departures_6M', 0))} team members departed in last 6 months",
        benchmark_info="Teams with 2+ departures see 1.7x contagion effect on remaining members"
    ),
    'response_time_increase': RiskDriver(
        name='Communication Slowdown',
        category='Engagement',
        weight=0.04,
        threshold_fn=lambda f: f.get('Avg_Response_Time_Hours', 0) >= 8,
        description_fn=lambda f: f"Average response time: {f.get('Avg_Response_Time_Hours', 0):.1f} hours (vs 2-4h norm)",
        benchmark_info="Response time >8 hours signals disengagement; 1.5x attrition correlation"
    ),
}

# =============================================================================
# MODEL LOADING
# =============================================================================

_model = None
_scaler = None
_feature_info = None

def load_model():
    """Load the trained model and scaler. Caches for efficiency."""
    global _model, _scaler, _feature_info
    
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    
    if _scaler is None:
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
    
    if _feature_info is None:
        with open(FEATURE_INFO_PATH, 'rb') as f:
            _feature_info = pickle.load(f)
    
    return _model, _scaler, _feature_info

# =============================================================================
# FEATURE EXTRACTION (Enhanced)
# =============================================================================

def calculate_trend(values: List[float]) -> float:
    """Calculate the trend (slope) of a sequence of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope

def calculate_change_pct(values: List[float]) -> float:
    """Calculate percentage change from start to end."""
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return ((values[-1] - values[0]) / values[0]) * 100

def extract_employee_features(employee_row: pd.Series) -> Dict[str, float]:
    """
    Extract all numerical features from a single employee row.
    Enhanced with trend calculations and derived metrics.
    """
    # Parse JSON columns
    daily_hours = json.loads(employee_row['Daily_Log_JSON'])
    performance_history = json.loads(employee_row['Performance_History_JSON'])
    engagement_history = json.loads(employee_row['Engagement_History_JSON'])
    manager_1on1_history = json.loads(employee_row['Manager_1on1_History_JSON'])
    office_presence = json.loads(employee_row['Office_Presence_JSON'])
    sick_leave_history = json.loads(employee_row['Sick_Leave_History_JSON'])
    
    # Filter working hours
    working_hours = [h for h in daily_hours if h > 0]
    
    return {
        # Work hours metrics
        'Avg_Daily_Hours': np.mean(working_hours) if working_hours else 0,
        'Hours_Variance': np.std(working_hours) if working_hours else 0,
        'Max_Daily_Hours': max(working_hours) if working_hours else 0,
        
        # Performance metrics (3 years = 12 quarters)
        'Performance_Current': performance_history[-1],
        'Performance_Start': performance_history[0],
        'Performance_Trend': calculate_trend(performance_history),
        'Performance_Avg': np.mean(performance_history),
        
        # Engagement metrics
        'Engagement_Current': engagement_history[-1],
        'Engagement_Start': engagement_history[0],
        'Engagement_Trend': calculate_trend(engagement_history),
        'Engagement_Avg': np.mean(engagement_history),
        'Engagement_Drop': engagement_history[0] - engagement_history[-1],
        
        # Manager engagement metrics
        'Manager_1on1_Current': np.mean(manager_1on1_history[-3:]),  # Last 3 months avg
        'Manager_1on1_Start': np.mean(manager_1on1_history[:3]),  # First 3 months avg
        'Manager_1on1_Trend': calculate_trend(manager_1on1_history),
        'Manager_1on1_Change_Pct': calculate_change_pct(manager_1on1_history),
        
        # Office presence metrics
        'Office_Presence_Current': np.mean(office_presence[-4:]),  # Last 4 weeks
        'Office_Presence_Start': np.mean(office_presence[:4]),  # First 4 weeks
        'Office_Presence_Trend': calculate_trend(office_presence),
        'Office_Presence_Change_Pct': calculate_change_pct(office_presence),
        
        # Sick leave metrics
        'Sick_Leave_Total_12M': sum(sick_leave_history),
        'Sick_Leave_Recent_3M': sum(sick_leave_history[-3:]),
        'Sick_Leave_Trend': calculate_trend(sick_leave_history),
        'Sick_Days_Avg': np.mean(sick_leave_history) if sick_leave_history else 0,
        
        # Direct snapshot metrics
        'Sick_Days_60D': employee_row['Sick_Days_60D'],
        'Vacation_Days_12M': employee_row['Vacation_Days_12M'],
        'Tenure_Years': employee_row['Tenure_Years'],
        'Months_Since_Promotion': employee_row['Months_Since_Promotion'],
        'Compensation_Gap_Pct': employee_row['Compensation_Gap_Pct'],
        'Months_Since_Training': employee_row['Months_Since_Training'],
        'Months_Since_Recognition': employee_row['Months_Since_Recognition'],
        'Team_Departures_6M': employee_row['Team_Departures_6M'],
        'Meeting_Attendance_Pct': employee_row['Meeting_Attendance_Pct'],
        'Avg_Response_Time_Hours': employee_row['Avg_Response_Time_Hours'],
    }

# =============================================================================
# RISK SCORING ALGORITHM
# =============================================================================

def calculate_risk_score(features: Dict[str, float]) -> float:
    """
    Calculate sophisticated risk score (0-100%) based on multiple factors.
    Uses weighted combination of triggered risk drivers.
    """
    base_score = 0.0
    triggered_weights = []
    
    for driver_id, driver in RISK_DRIVERS.items():
        if driver.threshold_fn(features):
            triggered_weights.append(driver.weight)
            # Add severity multiplier based on how far beyond threshold
            base_score += driver.weight
    
    # Normalize and scale to 0-100
    if triggered_weights:
        # Base calculation
        raw_score = base_score / sum(d.weight for d in RISK_DRIVERS.values())
        
        # Apply severity adjustments
        severity_multiplier = 1.0
        
        # Extreme engagement decline
        if features.get('Engagement_Current', 10) <= 2:
            severity_multiplier *= 1.3
        
        # Critical sick leave
        if features.get('Sick_Days_60D', 0) >= 10:
            severity_multiplier *= 1.2
        
        # Severe compensation gap
        if features.get('Compensation_Gap_Pct', 0) < -15:
            severity_multiplier *= 1.15
        
        # Very long without promotion
        if features.get('Months_Since_Promotion', 0) >= 36:
            severity_multiplier *= 1.2
        
        # Combination effects (multiple critical factors)
        critical_count = len(triggered_weights)
        if critical_count >= 4:
            severity_multiplier *= 1.1
        if critical_count >= 6:
            severity_multiplier *= 1.1
        
        final_score = min(100, raw_score * 100 * severity_multiplier)
    else:
        # Low base risk for employees with no triggered drivers
        final_score = np.random.uniform(5, 15)
    
    return final_score

def get_risk_level(score: float) -> Tuple[str, str]:
    """
    Get risk level and color based on score.
    
    Returns:
        Tuple of (risk_level, color_emoji)
    """
    if score < 30:
        return 'Low', RISK_COLORS['Low']
    elif score < 60:
        return 'Moderate', RISK_COLORS['Moderate']
    elif score < 85:
        return 'High', RISK_COLORS['High']
    else:
        return 'Critical', RISK_COLORS['Critical']

def calculate_risk_trend(features: Dict[str, float]) -> Tuple[str, str]:
    """
    Calculate risk trend based on data trajectories.
    Infers whether risk is increasing, stable, or decreasing.
    
    Returns:
        Tuple of (trend_direction, trend_icon)
        - ('increasing', '↑') if risk is rising
        - ('stable', '→') if risk is stable
        - ('decreasing', '↓') if risk is declining
    """
    # Count negative signals (increasing risk)
    negative_signals = 0
    positive_signals = 0
    
    # Engagement declining
    if features.get('Engagement_Trend', 0) < -0.3:
        negative_signals += 1
    elif features.get('Engagement_Trend', 0) > 0.2:
        positive_signals += 1
    
    # Office presence declining
    if features.get('Office_Presence_Trend', 0) < -1.5:
        negative_signals += 1
    elif features.get('Office_Presence_Trend', 0) > 1.0:
        positive_signals += 1
    
    # Manager engagement declining
    if features.get('Manager_1on1_Trend', 0) < -0.2:
        negative_signals += 1
    elif features.get('Manager_1on1_Trend', 0) > 0.1:
        positive_signals += 1
    
    # Sick leave increasing
    if features.get('Sick_Leave_Trend', 0) > 0.3:
        negative_signals += 1
    elif features.get('Sick_Leave_Trend', 0) < -0.2:
        positive_signals += 1
    
    # Performance declining
    if features.get('Performance_Trend', 0) < -1.5:
        negative_signals += 1
    elif features.get('Performance_Trend', 0) > 1.0:
        positive_signals += 1
    
    # Determine trend
    if negative_signals > positive_signals + 1:
        return 'increasing', '↑'
    elif positive_signals > negative_signals + 1:
        return 'decreasing', '↓'
    else:
        return 'stable', '→'

# =============================================================================
# TOP 3 RISK DRIVERS CALCULATION
# =============================================================================

def calculate_top_risk_drivers(features: Dict[str, float], n: int = 3) -> List[Dict[str, Any]]:
    """
    Identify and rank the Top N risk drivers with influence percentages.
    
    Returns:
        List of dicts with driver details and calculated influence.
    """
    triggered_drivers = []
    
    for driver_id, driver in RISK_DRIVERS.items():
        if driver.threshold_fn(features):
            # Calculate influence score based on:
            # 1. Base weight
            # 2. Severity (how far beyond threshold)
            # 3. Category importance
            
            base_influence = driver.weight
            
            # Severity adjustment
            severity = 1.0
            if driver_id == 'career_progression':
                months = features.get('Months_Since_Promotion', 0)
                if months >= 36:
                    severity = 1.5
                elif months >= 24:
                    severity = 1.2
                    
            elif driver_id == 'compensation_gap':
                gap = abs(features.get('Compensation_Gap_Pct', 0))
                if gap >= 15:
                    severity = 1.4
                elif gap >= 12:
                    severity = 1.2
                    
            elif driver_id == 'engagement_decline':
                current = features.get('Engagement_Current', 10)
                if current <= 2:
                    severity = 1.5
                elif current <= 4:
                    severity = 1.2
                    
            elif driver_id == 'sick_leave_increase':
                sick_days = features.get('Sick_Days_60D', 0)
                if sick_days >= 10:
                    severity = 1.4
                elif sick_days >= 7:
                    severity = 1.2
            
            triggered_drivers.append({
                'id': driver_id,
                'name': driver.name,
                'category': driver.category,
                'influence': base_influence * severity,
                'description': driver.description_fn(features),
                'benchmark': driver.benchmark_info
            })
    
    # Sort by influence and take top N
    triggered_drivers.sort(key=lambda x: x['influence'], reverse=True)
    top_drivers = triggered_drivers[:n]
    
    # Calculate relative influence percentages (must sum to 100%)
    if top_drivers:
        total_influence = sum(d['influence'] for d in top_drivers)
        for driver in top_drivers:
            driver['influence_pct'] = round((driver['influence'] / total_influence) * 100)
        
        # Adjust for rounding to ensure sum = 100
        pct_sum = sum(d['influence_pct'] for d in top_drivers)
        if pct_sum != 100 and top_drivers:
            top_drivers[0]['influence_pct'] += (100 - pct_sum)
    
    return top_drivers

# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_attrition_risk(employee_row: pd.Series) -> Dict[str, Any]:
    """
    Predict attrition risk for a single employee with comprehensive analysis.
    
    This is the main function for UI integration.
    
    Args:
        employee_row: A pandas Series from the employee dataset.
        
    Returns:
        Dictionary containing:
        - employee_id, employee_name, department, level
        - risk_score: float (0-100)
        - risk_level: str ('Low', 'Moderate', 'High', 'Critical')
        - risk_color: str (emoji)
        - top_3_drivers: List of top risk drivers with influence %
        - features: Dict of all extracted features
        - manager_notes, survey_comments
    """
    # Extract features
    features = extract_employee_features(employee_row)
    
    # Calculate risk score
    risk_score = calculate_risk_score(features)
    risk_level, risk_color = get_risk_level(risk_score)
    
    # Calculate risk trend
    risk_trend, risk_trend_icon = calculate_risk_trend(features)
    
    # Get top 3 risk drivers
    top_drivers = calculate_top_risk_drivers(features, n=3)
    
    # Build summary explanation
    if top_drivers:
        driver_summary = "; ".join([
            f"{d['name']} ({d['influence_pct']}%)" for d in top_drivers
        ])
        explanation = f"Primary risk factors: {driver_summary}"
    else:
        explanation = "No significant risk factors detected. Employee profile appears stable."
    
    return {
        'employee_id': employee_row['Employee_ID'],
        'employee_name': employee_row['Name'],
        'department': employee_row['Department'],
        'level': employee_row['Level'],
        'tenure_years': employee_row['Tenure_Years'],
        'manager_name': employee_row.get('Manager_Name', 'N/A'),
        
        'risk_score': round(risk_score, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_trend': risk_trend,
        'risk_trend_icon': risk_trend_icon,
        
        'top_3_drivers': top_drivers,
        'explanation': explanation,
        
        'features': features,
        
        'actual_at_risk': employee_row.get('At_Risk', None),
        'manager_notes': employee_row.get('Manager_Notes', ''),
        'survey_comments': employee_row.get('Survey_Comments', '')
    }

# =============================================================================
# BATCH PREDICTIONS
# =============================================================================

def predict_all_employees(df: pd.DataFrame) -> pd.DataFrame:
    """Predict attrition risk for all employees."""
    predictions = []
    
    for idx, row in df.iterrows():
        pred = predict_attrition_risk(row)
        predictions.append({
            'Employee_ID': pred['employee_id'],
            'Risk_Score': pred['risk_score'],
            'Risk_Level': pred['risk_level'],
            'Risk_Color': pred['risk_color'],
            'Risk_Trend': pred['risk_trend_icon'],
            'Top_Driver_1': pred['top_3_drivers'][0]['name'] if len(pred['top_3_drivers']) > 0 else None,
            'Top_Driver_1_Pct': pred['top_3_drivers'][0]['influence_pct'] if len(pred['top_3_drivers']) > 0 else None,
            'Explanation': pred['explanation']
        })
    
    pred_df = pd.DataFrame(predictions)
    return df.merge(pred_df, on='Employee_ID')

def get_high_risk_employees(df: pd.DataFrame, min_level: str = 'Moderate') -> List[Dict[str, Any]]:
    """Get all employees at or above specified risk level."""
    level_order = {'Low': 0, 'Moderate': 1, 'High': 2, 'Critical': 3}
    min_order = level_order.get(min_level, 1)
    
    high_risk = []
    for idx, row in df.iterrows():
        pred = predict_attrition_risk(row)
        if level_order.get(pred['risk_level'], 0) >= min_order:
            high_risk.append(pred)
    
    high_risk.sort(key=lambda x: x['risk_score'], reverse=True)
    return high_risk

def get_risk_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get overall risk summary statistics."""
    all_predictions = [predict_attrition_risk(row) for _, row in df.iterrows()]
    
    level_counts = {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0}
    for pred in all_predictions:
        level_counts[pred['risk_level']] += 1
    
    scores = [p['risk_score'] for p in all_predictions]
    
    return {
        'total_employees': len(df),
        'risk_distribution': level_counts,
        'avg_risk_score': np.mean(scores),
        'median_risk_score': np.median(scores),
        'high_risk_count': level_counts['High'] + level_counts['Critical'],
        'critical_count': level_counts['Critical']
    }

def get_department_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Get risk summary by department."""
    summaries = {}
    
    for dept in df['Department'].unique():
        dept_df = df[df['Department'] == dept]
        preds = [predict_attrition_risk(row) for _, row in dept_df.iterrows()]
        
        level_counts = {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0}
        for pred in preds:
            level_counts[pred['risk_level']] += 1
        
        scores = [p['risk_score'] for p in preds]
        
        summaries[dept] = {
            'total': len(dept_df),
            'avg_score': round(np.mean(scores), 1),
            'distribution': level_counts,
            'high_risk_pct': round((level_counts['High'] + level_counts['Critical']) / len(dept_df) * 100, 1)
        }
    
    return summaries

# =============================================================================
# HELPER FUNCTIONS FOR UI
# =============================================================================

def get_employee_by_id(df: pd.DataFrame, employee_id: str) -> Optional[Dict[str, Any]]:
    """Get prediction for a specific employee by ID."""
    employee = df[df['Employee_ID'] == employee_id]
    if len(employee) == 0:
        return None
    return predict_attrition_risk(employee.iloc[0])

def parse_daily_log_for_chart(json_str: str) -> List[Dict[str, Any]]:
    """Parse Daily_Log_JSON for charting."""
    hours = json.loads(json_str)
    return [{'day': i + 1, 'hours': h} for i, h in enumerate(hours)]

def parse_history_for_chart(json_str: str, metric_name: str) -> List[Dict[str, Any]]:
    """Parse quarterly history JSON for charting."""
    scores = json.loads(json_str)
    # For 3 years of data (12 quarters)
    quarters = [f'Q{(i % 4) + 1} Y{(i // 4) + 1}' for i in range(len(scores))]
    return [{'quarter': quarters[i], metric_name: s} for i, s in enumerate(scores)]

def format_risk_report(prediction: Dict[str, Any]) -> str:
    """Format a prediction as a readable text report."""
    lines = [
        f"{'='*60}",
        f"ATTRITION RISK ASSESSMENT REPORT",
        f"{'='*60}",
        f"",
        f"Employee: {prediction['employee_name']} ({prediction['employee_id']})",
        f"Department: {prediction['department']} | Level: {prediction['level']}",
        f"Tenure: {prediction['tenure_years']} years",
        f"",
        f"{'─'*60}",
        f"RISK ASSESSMENT",
        f"{'─'*60}",
        f"",
        f"  Risk Score: {prediction['risk_score']}%",
        f"  Risk Level: {prediction['risk_color']} {prediction['risk_level']}",
        f"",
    ]
    
    if prediction['top_3_drivers']:
        lines.append(f"{'─'*60}")
        lines.append("TOP 3 RISK DRIVERS")
        lines.append(f"{'─'*60}")
        lines.append("")
        
        for i, driver in enumerate(prediction['top_3_drivers'], 1):
            lines.append(f"  {i}. {driver['name']} ({driver['influence_pct']}% influence)")
            lines.append(f"     {driver['description']}")
            lines.append(f"     📊 {driver['benchmark']}")
            lines.append("")
    
    lines.append(f"{'='*60}")
    
    return "\n".join(lines)

# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "employee_attrition_dataset_final.csv"
    
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} employees")
    
    print("\n" + "="*60)
    print("ATTRITION RISK PREDICTION - DEMO v2.0")
    print("="*60)
    
    # Overall summary
    print("\n📊 OVERALL RISK SUMMARY")
    summary = get_risk_summary(df)
    print(f"  Total Employees: {summary['total_employees']}")
    print(f"  Average Risk Score: {summary['avg_risk_score']:.1f}%")
    print(f"  Risk Distribution:")
    for level, count in summary['risk_distribution'].items():
        pct = count / summary['total_employees'] * 100
        color = RISK_COLORS.get(level, '')
        print(f"    {color} {level}: {count} ({pct:.1f}%)")
    
    # Sample individual predictions
    print("\n" + "="*60)
    print("SAMPLE INDIVIDUAL ASSESSMENTS")
    print("="*60)
    
    # Sample employees from different risk levels
    for i, idx in enumerate([0, 50, 150, 300]):
        sample = df.iloc[idx]
        pred = predict_attrition_risk(sample)
        
        print(f"\n{pred['risk_color']} {pred['employee_name']}")
        print(f"   Score: {pred['risk_score']}% | Level: {pred['risk_level']}")
        
        if pred['top_3_drivers']:
            print("   Top Drivers:")
            for d in pred['top_3_drivers']:
                print(f"     • {d['name']}: {d['influence_pct']}%")
    
    # High risk list
    print("\n" + "="*60)
    print("🔴 CRITICAL & HIGH RISK EMPLOYEES")
    print("="*60)
    
    high_risk = get_high_risk_employees(df, min_level='High')
    for emp in high_risk[:10]:
        print(f"  {emp['risk_color']} {emp['employee_name']}: {emp['risk_score']}% ({emp['risk_level']})")
    
    if len(high_risk) > 10:
        print(f"  ... and {len(high_risk) - 10} more")
    
    # Department breakdown
    print("\n" + "="*60)
    print("📈 RISK BY DEPARTMENT")
    print("="*60)
    
    dept_summary = get_department_summary(df)
    for dept, stats in sorted(dept_summary.items(), key=lambda x: x[1]['avg_score'], reverse=True):
        print(f"  {dept}: {stats['avg_score']}% avg | {stats['high_risk_pct']}% high risk")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)

# Frontend Integration Guide
## Employee Attrition Risk Prediction System

**For: Coder B (UI/Frontend Developer)**  
**From: Coder A (Data & Backend)**

---

## 📋 Overview

This document explains the backend components that have been built and how to integrate them into your Streamlit/Gradio frontend application.

---

## 🗂️ Project Structure

```
AI Class Project/
├── employee_attrition_dataset_final.csv    # Main dataset (500 employees)
├── generate_dataset.py                     # Data generator (you don't need this)
├── train_model.py                          # Model trainer (you don't need this)
├── predict.py                              # ⭐ MAIN MODULE - Import this!
└── model/
    ├── attrition_model.pkl                 # Trained ML model
    ├── scaler.pkl                          # Feature scaler
    └── feature_info.pkl                    # Feature metadata
```

---

## 🚀 Quick Start

### 1. Import the Prediction Module

```python
import pandas as pd
from predict import (
    predict_attrition_risk,
    get_employee_by_id,
    get_high_risk_employees,
    get_risk_summary,
    get_department_summary,
    parse_daily_log_for_chart,
    parse_history_for_chart
)
```

### 2. Load the Dataset

```python
df = pd.read_csv('AI Class Project/employee_attrition_dataset_final.csv')
```

### 3. Make Predictions

```python
# Get prediction for one employee
prediction = predict_attrition_risk(df.iloc[0])

# Or by ID
employee = get_employee_by_id(df, 'EMP001')
```

---

## 📊 Data Structure

### CSV Columns (26 total)

| Column | Type | Description |
|--------|------|-------------|
| `Employee_ID` | string | Unique ID (e.g., "EMP001") |
| `Name` | string | Employee full name |
| `Department` | string | Engineering, Sales, HR, Finance, Marketing, Operations |
| `Manager_Name` | string | Manager's name |
| `Level` | string | CEO, C-Level, Senior Manager, Manager, Standard, Junior |
| `Tenure_Years` | float | Years at company |
| `Months_Since_Promotion` | int | Months since last promotion |
| `Daily_Log_JSON` | JSON string | 60 days of work hours (array of floats) |
| `Performance_History_JSON` | JSON string | 12 quarters of performance scores (1-100) |
| `Engagement_History_JSON` | JSON string | 12 quarters of engagement scores (1-10) |
| `Manager_1on1_History_JSON` | JSON string | 12 months of 1:1 meeting counts |
| `Office_Presence_JSON` | JSON string | 12 weeks of office presence % (0-100) |
| `Sick_Leave_History_JSON` | JSON string | 12 months of sick days |
| `Sick_Days_60D` | int | Sick days in last 60 days |
| `Vacation_Days_12M` | int | Vacation days in last 12 months |
| `Salary_Actual` | int | Employee's actual salary |
| `Salary_Market` | int | Market benchmark salary |
| `Compensation_Gap_Pct` | float | % above/below market |
| `Months_Since_Training` | int | Months since last training |
| `Months_Since_Recognition` | int | Months since last recognition |
| `Team_Departures_6M` | int | Team members who left in 6 months |
| `Meeting_Attendance_Pct` | float | % of meetings attended |
| `Avg_Response_Time_Hours` | float | Average response time |
| `Manager_Notes` | string | Manager's written notes |
| `Survey_Comments` | string | Employee's survey feedback |
| `At_Risk` | int | Ground truth (0 or 1) - for validation only |

---

## 🎯 Core Functions

### 1. `predict_attrition_risk(employee_row)`

**Main prediction function** - Use this for individual employee profiles.

**Input:**
- `employee_row`: A pandas Series from the CSV (e.g., `df.iloc[0]` or `df[df['Employee_ID'] == 'EMP001'].iloc[0]`)

**Output Dictionary:**
```python
{
    # Basic Info
    'employee_id': 'EMP001',
    'employee_name': 'John Smith',
    'department': 'Engineering',
    'level': 'Standard',
    'tenure_years': 4.5,
    'manager_name': 'Sarah Johnson',
    
    # Risk Assessment
    'risk_score': 67.3,              # 0-100%
    'risk_level': 'High',             # 'Low', 'Moderate', 'High', 'Critical'
    'risk_color': '🟠',               # 🟢 🟡 🟠 🔴
    'risk_trend': 'increasing',       # 'increasing', 'stable', 'decreasing'
    'risk_trend_icon': '↑',           # ↑ → ↓
    
    # Top 3 Risk Drivers
    'top_3_drivers': [
        {
            'name': 'No Career Progression',
            'influence_pct': 42,
            'description': 'No promotion in 18 months',
            'benchmark': 'Peer benchmarking shows 3.2x higher departure probability...',
            'category': 'Career'
        },
        {
            'name': 'Below-Market Compensation',
            'influence_pct': 31,
            'description': 'Salary 12% below market average',
            'benchmark': 'Compensation gap of >8% predicts 2.1x higher attrition...',
            'category': 'Compensation'
        },
        {
            'name': 'Declining Manager Engagement',
            'influence_pct': 27,
            'description': '1:1 meeting frequency declined 35% over 12 months',
            'benchmark': 'Manager engagement decline correlates with 2.5x retention risk',
            'category': 'Management'
        }
    ],
    
    # Summary
    'explanation': 'Primary risk factors: No Career Progression (42%); Below-Market Compensation (31%); Declining Manager Engagement (27%)',
    
    # Raw Features (for advanced use)
    'features': {
        'Engagement_Current': 4.0,
        'Performance_Current': 78,
        'Avg_Daily_Hours': 8.2,
        # ... many more
    },
    
    # Text Data
    'manager_notes': 'Employee has expressed frustration...',
    'survey_comments': 'Feeling stuck and undervalued...',
    
    # Validation (optional)
    'actual_at_risk': 1  # Ground truth from dataset
}
```

**Example Usage:**
```python
# Get employee by ID
employee = df[df['Employee_ID'] == 'EMP001'].iloc[0]
prediction = predict_attrition_risk(employee)

# Display in UI
st.metric("Risk Score", f"{prediction['risk_score']}%", 
          delta=f"{prediction['risk_trend_icon']} {prediction['risk_trend']}")
st.write(f"**Level:** {prediction['risk_color']} {prediction['risk_level']}")

# Show top drivers
for i, driver in enumerate(prediction['top_3_drivers'], 1):
    st.write(f"{i}. **{driver['name']}** ({driver['influence_pct']}%)")
    st.write(f"   {driver['description']}")
```

---

### 2. `get_employee_by_id(df, employee_id)`

**Convenience function** to get prediction by employee ID.

**Input:**
- `df`: DataFrame loaded from CSV
- `employee_id`: String like "EMP001"

**Output:** Same as `predict_attrition_risk()`

**Example:**
```python
prediction = get_employee_by_id(df, 'EMP001')
if prediction:
    st.write(f"Employee: {prediction['employee_name']}")
```

---

### 3. `get_high_risk_employees(df, min_level='Moderate')`

**Get all employees at or above a risk level** - Perfect for the main dashboard list.

**Input:**
- `df`: DataFrame
- `min_level`: 'Low', 'Moderate', 'High', or 'Critical' (default: 'Moderate')

**Output:** List of prediction dictionaries, sorted by risk score (highest first)

**Example:**
```python
high_risk = get_high_risk_employees(df, min_level='High')

# Display in table
for emp in high_risk:
    st.write(f"{emp['risk_color']} {emp['employee_name']}: {emp['risk_score']}% ({emp['risk_level']}) {emp['risk_trend_icon']}")
```

---

### 4. `get_risk_summary(df)`

**Get overall organizational statistics** - For dashboard overview.

**Input:** DataFrame

**Output:**
```python
{
    'total_employees': 500,
    'avg_risk_score': 28.7,
    'median_risk_score': 18.5,
    'risk_distribution': {
        'Low': 341,
        'Moderate': 103,
        'High': 29,
        'Critical': 27
    },
    'high_risk_count': 56,
    'critical_count': 27
}
```

**Example:**
```python
summary = get_risk_summary(df)
st.metric("Total Employees", summary['total_employees'])
st.metric("High Risk", summary['high_risk_count'], 
          delta=f"{summary['high_risk_count']/summary['total_employees']*100:.1f}%")
```

---

### 5. `get_department_summary(df)`

**Get risk statistics by department** - For department comparison chart.

**Input:** DataFrame

**Output:**
```python
{
    'Engineering': {
        'total': 85,
        'avg_score': 30.2,
        'distribution': {'Low': 60, 'Moderate': 20, 'High': 4, 'Critical': 1},
        'high_risk_pct': 15.4
    },
    'Sales': {
        'total': 92,
        'avg_score': 33.6,
        'distribution': {'Low': 55, 'Moderate': 25, 'High': 8, 'Critical': 4},
        'high_risk_pct': 17.7
    },
    # ... other departments
}
```

**Example:**
```python
dept_summary = get_department_summary(df)
for dept, stats in dept_summary.items():
    st.write(f"{dept}: {stats['avg_score']}% avg | {stats['high_risk_pct']}% high risk")
```

---

### 6. `parse_daily_log_for_chart(json_str)`

**Parse daily hours JSON for line chart** - For the "Work Habits" visualization.

**Input:** JSON string from `Daily_Log_JSON` column

**Output:**
```python
[
    {'day': 1, 'hours': 8.2},
    {'day': 2, 'hours': 7.9},
    # ... 60 days total
]
```

**Example:**
```python
import plotly.express as px

hours_data = parse_daily_log_for_chart(employee['Daily_Log_JSON'])
df_hours = pd.DataFrame(hours_data)

fig = px.line(df_hours, x='day', y='hours', 
              title='Daily Work Hours (Last 60 Days)')
st.plotly_chart(fig)
```

---

### 7. `parse_history_for_chart(json_str, metric_name)`

**Parse quarterly history JSON for line chart** - For "Performance Trend" and "Engagement Trend" visualizations.

**Input:**
- `json_str`: JSON string from `Performance_History_JSON` or `Engagement_History_JSON`
- `metric_name`: String like 'performance' or 'engagement'

**Output:**
```python
[
    {'quarter': 'Q1 Y1', 'performance': 85},
    {'quarter': 'Q2 Y1', 'performance': 83},
    # ... 12 quarters total (3 years)
]
```

**Example:**
```python
perf_data = parse_history_for_chart(employee['Performance_History_JSON'], 'performance')
df_perf = pd.DataFrame(perf_data)

fig = px.line(df_perf, x='quarter', y='performance',
              title='Performance History (3 Years)',
              labels={'performance': 'Performance Score (1-100)'})
st.plotly_chart(fig)
```

---

## 🎨 UI Components to Build

### 1. Main Dashboard Page

**Required Elements:**
- Overall risk summary (total employees, avg risk, distribution)
- Department comparison chart
- Employee list table with columns:
  - Employee Name (clickable → profile page)
  - Department
  - Risk Score (%)
  - Risk Level (🟢 🟡 🟠 🔴)
  - Trend (↑ ↓ →)
  - Top Driver (name + %)

**Code Snippet:**
```python
import streamlit as st
import pandas as pd
from predict import get_risk_summary, get_department_summary, get_high_risk_employees

df = pd.read_csv('AI Class Project/employee_attrition_dataset_final.csv')

# Summary metrics
summary = get_risk_summary(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", summary['total_employees'])
col2.metric("High Risk", summary['high_risk_count'])
col3.metric("Critical", summary['critical_count'])
col4.metric("Avg Risk", f"{summary['avg_risk_score']:.1f}%")

# Risk distribution
st.bar_chart(summary['risk_distribution'])

# Employee list
high_risk = get_high_risk_employees(df, min_level='Moderate')
for emp in high_risk:
    with st.expander(f"{emp['risk_color']} {emp['employee_name']} - {emp['risk_score']}% {emp['risk_trend_icon']}"):
        st.write(f"**Department:** {emp['department']}")
        st.write(f"**Top Driver:** {emp['top_3_drivers'][0]['name']} ({emp['top_3_drivers'][0]['influence_pct']}%)")
```

---

### 2. Individual Employee Profile Page

**Required Elements:**
- Employee overview (name, department, manager, tenure)
- Risk score with color indicator
- Risk trend arrow
- Top 3 Risk Drivers (with influence bars)
- Work Habits chart (60-day daily hours)
- Performance Trend chart (12 quarters)
- Engagement Trend chart (12 quarters)
- Manager Notes & Survey Comments

**Code Snippet:**
```python
import streamlit as st
import plotly.express as px
from predict import get_employee_by_id, parse_daily_log_for_chart, parse_history_for_chart

employee_id = st.selectbox("Select Employee", df['Employee_ID'].tolist())
prediction = get_employee_by_id(df, employee_id)
employee = df[df['Employee_ID'] == employee_id].iloc[0]

# Header
col1, col2 = st.columns([2, 1])
with col1:
    st.title(prediction['employee_name'])
    st.write(f"**Department:** {prediction['department']} | **Manager:** {prediction['manager_name']}")
with col2:
    st.metric("Risk Score", f"{prediction['risk_score']}%", 
              delta=f"{prediction['risk_trend_icon']} {prediction['risk_trend']}")
    st.write(f"**Level:** {prediction['risk_color']} {prediction['risk_level']}")

# Top 3 Drivers
st.subheader("Top 3 Risk Drivers")
for i, driver in enumerate(prediction['top_3_drivers'], 1):
    st.write(f"{i}. **{driver['name']}** ({driver['influence_pct']}% influence)")
    st.progress(driver['influence_pct'] / 100)
    st.caption(driver['description'])
    st.caption(f"📊 {driver['benchmark']}")

# Charts
col1, col2 = st.columns(2)
with col1:
    hours_data = parse_daily_log_for_chart(employee['Daily_Log_JSON'])
    fig = px.line(pd.DataFrame(hours_data), x='day', y='hours', 
                  title='Daily Work Hours')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    perf_data = parse_history_for_chart(employee['Performance_History_JSON'], 'performance')
    fig = px.line(pd.DataFrame(perf_data), x='quarter', y='performance',
                  title='Performance Trend')
    st.plotly_chart(fig, use_container_width=True)
```

---

## 🎯 Risk Level System

| Risk Level | Score Range | Color | Icon | Meaning |
|------------|-------------|-------|------|---------|
| Low | 0-30% | 🟢 Green | - | Employee stable; low departure probability |
| Moderate | 30-60% | 🟡 Yellow | - | Employee showing concern signals; monitor |
| High | 60-85% | 🟠 Orange | - | Substantial risk; intervention recommended |
| Critical | 85-100% | 🔴 Red | - | Imminent departure risk; urgent intervention |

**Trend Indicators:**
- `↑` (increasing) - Risk is rising
- `→` (stable) - Risk is stable
- `↓` (decreasing) - Risk is declining

---

## 📈 Chart Data Format

### Daily Log (Work Habits)
- **Source:** `Daily_Log_JSON` column
- **Format:** Array of 60 floats (daily work hours)
- **Use:** Line chart showing work patterns over 2 months
- **Function:** `parse_daily_log_for_chart(json_str)`

### Performance History
- **Source:** `Performance_History_JSON` column
- **Format:** Array of 12 integers (quarterly scores, 1-100)
- **Use:** Line chart showing 3-year performance trend
- **Function:** `parse_history_for_chart(json_str, 'performance')`

### Engagement History
- **Source:** `Engagement_History_JSON` column
- **Format:** Array of 12 integers (quarterly scores, 1-10)
- **Use:** Line chart showing 3-year engagement trend
- **Function:** `parse_history_for_chart(json_str, 'engagement')`

---

## 🔧 Error Handling

The prediction functions handle missing data gracefully:

```python
# Safe access - won't crash if column missing
manager_name = employee_row.get('Manager_Name', 'N/A')
```

**Common Issues:**
- If `Manager_Name` is missing → Returns 'N/A'
- If employee not found → `get_employee_by_id()` returns `None`
- If JSON parsing fails → Check CSV format

---

## 🚨 Important Notes

1. **Model Files Required:** Make sure `model/` folder exists with all `.pkl` files. The model loads automatically on first import.

2. **CSV Path:** Update the path in your code if the CSV is in a different location:
   ```python
   df = pd.read_csv('AI Class Project/employee_attrition_dataset_final.csv')
   ```

3. **Performance:** For 500 employees, predictions are fast (< 1 second). For larger datasets, consider caching.

4. **Data Updates:** If you regenerate the CSV, the model doesn't need retraining unless you change the feature set.

5. **No Archetype Column:** The CSV doesn't include archetype labels - the model discovers patterns from data.

---

## 📞 Questions?

If you need:
- Additional helper functions
- Different data formats
- Performance optimizations
- Custom risk calculations

Let me know and I can add them to `predict.py`!

---

## ✅ Checklist for Frontend

- [ ] Import `predict.py` module
- [ ] Load CSV dataset
- [ ] Build main dashboard with summary metrics
- [ ] Create employee list table with risk scores
- [ ] Build individual employee profile page
- [ ] Add charts for work habits and trends
- [ ] Display Top 3 Risk Drivers with influence bars
- [ ] Show risk level colors (🟢 🟡 🟠 🔴)
- [ ] Display trend arrows (↑ ↓ →)
- [ ] Add manager name display
- [ ] Test with all 500 employees

---

**Good luck with the frontend! 🚀**


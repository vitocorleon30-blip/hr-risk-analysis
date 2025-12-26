# 📊 Employee Attrition Risk Prediction System - Complete Documentation

## Table of Contents
1. [How the App Works](#how-the-app-works)
2. [Individual Employee Profile](#individual-employee-profile)
3. [Complete App Documentation](#complete-app-documentation)

---

## 🚀 How the App Works

### Overview
The Employee Attrition Risk Prediction System is an enterprise-grade HR analytics dashboard that uses **machine learning and rule-based algorithms** to predict which employees are most likely to voluntarily leave the organization. The system processes comprehensive employee data, analyzes multiple risk factors, and provides actionable insights to help HR teams proactively address retention challenges.

### 📥 Inputs

The app processes the following data inputs:

| Input Type | Description | Source |
|------------|-------------|--------|
| **CSV Dataset** | `employee_attrition_dataset_final.csv` | Primary data source containing employee records |
| **Employee Attributes** | Demographics, tenure, role, department, compensation | CSV columns |
| **Historical JSON Data** | Daily work hours, performance scores, engagement metrics | JSON columns in CSV |
| **Behavioral Metrics** | Office presence, sick leave, manager 1-on-1 frequency | Calculated from JSON data |
| **Qualitative Data** | Manager notes, survey comments, feedback | Text fields in CSV |

#### Key Data Fields Processed:
- **Personal**: Age, Gender, Education, Marital Status
- **Employment**: Department, Job Role, Level, Tenure, Manager
- **Compensation**: Monthly Income, Salary Hike, Stock Options
- **Performance**: Performance Rating, Performance History (12 quarters)
- **Engagement**: Engagement History (12 quarters), Work-Life Balance
- **Behavioral**: Daily Work Hours (30 days), Office Presence, Sick Leave
- **Manager Relations**: Manager 1-on-1 History, Meeting Attendance
- **Career**: Months Since Promotion, Training History, Recognition

### 🔄 Data Processing Method

The app uses a **multi-stage processing pipeline**:

#### Stage 1: Feature Extraction (`predict.py`)
```python
extract_employee_features(employee_row)
```

**Process:**
1. **JSON Parsing**: Extracts and parses JSON columns containing historical data
   - `Daily_Log_JSON` → Daily work hours (30 days)
   - `Performance_History_JSON` → Quarterly performance scores (12 quarters)
   - `Engagement_History_JSON` → Quarterly engagement scores (12 quarters)
   - `Manager_1on1_History_JSON` → Monthly 1-on-1 frequency
   - `Office_Presence_JSON` → Weekly office attendance percentages
   - `Sick_Leave_History_JSON` → Monthly sick leave days

2. **Trend Calculation**: Computes statistical trends from historical data
   - **Linear Regression**: Calculates slope/trend direction (increasing/decreasing)
   - **Percentage Change**: Measures change from start to current period
   - **Averages**: Computes mean values for comparison

3. **Derived Metrics**: Creates composite indicators
   - `Engagement_Drop`: Difference between start and current engagement
   - `Manager_1on1_Change_Pct`: Percentage change in meeting frequency
   - `Office_Presence_Trend`: Trend in office attendance
   - `Sick_Leave_Trend`: Trend in sick leave patterns

#### Stage 2: Risk Driver Detection
The system evaluates **10 predefined risk drivers** with weighted influence:

| Risk Driver | Weight | Threshold | Description |
|-------------|--------|-----------|-------------|
| 🚫 **No Career Progression** | 18% | 18+ months since promotion | Stagnation in role advancement |
| 💰 **Below-Market Compensation** | 16% | 8%+ below market average | Salary gap vs. industry standards |
| 👥 **Declining Manager Engagement** | 14% | 30%+ decline in 1-on-1s | Reduced manager interaction |
| 🏢 **Declining Office Presence** | 12% | 20%+ drop in attendance | Reduced physical presence |
| 🏥 **Increasing Sick Leave** | 11% | 5+ days in 60 days | Health/burnout indicators |
| 📉 **Declining Engagement Score** | 10% | 3+ point drop or ≤4/10 | Overall engagement decline |
| 📚 **Training & Development Gap** | 6% | 12+ months without training | Lack of skill development |
| 🏆 **Recognition Gap** | 5% | 8+ months without recognition | Lack of acknowledgment |
| 👨‍👩‍👧‍👦 **Team Turnover Contagion** | 4% | 2+ team departures in 6M | Peer departure effects |
| 📧 **Communication Slowdown** | 4% | 8+ hour avg response time | Disengagement signals |

#### Stage 3: Risk Score Calculation
```python
calculate_risk_score(features)
```

**Algorithm:**
1. **Trigger Detection**: Checks which risk drivers are activated based on thresholds
2. **Weighted Sum**: Calculates base score from triggered driver weights
3. **Severity Multipliers**: Applies multipliers for extreme cases:
   - Engagement ≤ 2/10: ×1.3
   - Sick days ≥ 10 in 60 days: ×1.2
   - Compensation gap ≥ 15%: ×1.15
   - No promotion ≥ 36 months: ×1.2
   - 4+ drivers triggered: ×1.1
   - 6+ drivers triggered: ×1.1 (additional)
4. **Normalization**: Scales to 0-100% probability
5. **Risk Level Assignment**:
   - 🟢 **Low**: 0-30%
   - 🟡 **Moderate**: 30-60%
   - 🟠 **High**: 60-85%
   - 🔴 **Critical**: 85-100%

#### Stage 4: Top 3 Risk Drivers Ranking
```python
calculate_top_risk_drivers(features, n=3)
```

**Process:**
1. Identifies all triggered risk drivers
2. Calculates influence score (base weight × severity multiplier)
3. Ranks by influence
4. Calculates relative percentages (must sum to 100%)
5. Returns top 3 with descriptions and benchmarks

### 🤖 How the Model Predicts Employee Quitting

The prediction system uses a **hybrid approach** combining:

#### 1. **Rule-Based Risk Assessment**
- **10 Risk Drivers**: Each with specific thresholds based on HR research
- **Weighted Scoring**: Drivers have different influence weights (4-18%)
- **Severity Adjustments**: Extreme cases get multipliers
- **Trend Analysis**: Historical patterns indicate direction

#### 2. **Statistical Pattern Recognition**
- **Trend Detection**: Linear regression on historical metrics
- **Anomaly Detection**: Identifies deviations from normal patterns
- **Correlation Analysis**: Links multiple risk factors together

#### 3. **Behavioral Indicators**
The model analyzes behavioral signals:
- **Engagement Decline**: Dropping satisfaction scores
- **Reduced Interaction**: Fewer manager meetings, slower responses
- **Physical Disengagement**: Less office presence
- **Health Indicators**: Increased sick leave (potential burnout)

#### 4. **Career Progression Signals**
- **Stagnation**: Long periods without promotion/training
- **Compensation Gaps**: Below-market pay
- **Recognition**: Lack of acknowledgment

#### 5. **Social Contagion Effects**
- **Team Turnover**: Multiple departures in same team
- **Department Patterns**: Systemic issues in specific departments

**Prediction Formula:**
```
Risk Score = Σ(Triggered Driver Weights) × Severity Multipliers × 100
```

The score represents the **probabilistic likelihood** (0-100%) that an employee will voluntarily leave within the next 12 months.

### 💵 How Estimated Attrition Cost is Calculated

The app calculates financial risk using a **cost-per-risk-level model**:

```python
estimated_financial_risk = (critical_count × $30,000) + (high_risk_count × $15,000)
```

#### Cost Model Breakdown:

| Risk Level | Count | Cost per Employee | Rationale |
|------------|-------|-------------------|-----------|
| 🔴 **Critical** (85-100%) | `critical_count` | **$30,000** | Immediate departure risk - includes recruitment, onboarding, lost productivity, knowledge transfer |
| 🟠 **High** (60-85%) | `high_risk_count` | **$15,000** | Elevated risk - proactive intervention costs, potential replacement preparation |

#### Cost Components Included:
1. **Recruitment Costs**: Job postings, agency fees, screening
2. **Onboarding**: Training, orientation, ramp-up time
3. **Lost Productivity**: Time to find replacement + learning curve
4. **Knowledge Transfer**: Documentation, handover processes
5. **Intervention Costs**: Retention efforts, salary adjustments, development programs

#### Calculation Example:
```
If you have:
- 5 Critical Risk employees → 5 × $30,000 = $150,000
- 12 High Risk employees → 12 × $15,000 = $180,000
Total Estimated Cost = $330,000
```

**Note**: This is a conservative estimate. Actual costs can vary significantly based on:
- Employee role and seniority
- Industry and market conditions
- Time to fill positions
- Additional hidden costs (team morale, project delays)

### 📤 Outputs

The app generates multiple types of outputs:

#### 1. **Dashboard Metrics**
- Total employees count
- High-risk employee count
- Critical-risk employee count
- Estimated attrition cost (formatted as currency)

#### 2. **Visualizations**
- **Risk Distribution Pie Chart**: Shows percentage of employees in each risk category
- **Department Risk Bar Chart**: Average risk score by department
- **Risk Composition Stacked Bar**: Breakdown of risk levels within departments
- **Attrition Risk Curve**: Risk vs. tenure relationship
- **Job Level Risk Chart**: Risk distribution by seniority

#### 3. **Employee Risk Profiles**
- Risk score (0-100%)
- Risk level (Low/Moderate/High/Critical)
- Top 3 risk drivers with influence percentages
- Risk trend (increasing/stable/decreasing)
- Detailed feature breakdown

#### 4. **Department Analysis**
- Average risk score per department
- High-risk percentage per department
- Employee lists with risk scores
- Risk vs. tenure scatter plots

#### 5. **AI Assistant Responses**
- Natural language answers to HR questions
- Data-driven recommendations
- Pattern analysis insights
- Interactive charts on demand

---

## 👤 Individual Employee Profile

### Overview
The Individual Employee Profile provides a **comprehensive, visually-rich view** of each employee's risk assessment, personal information, performance history, and qualitative feedback. It's designed for HR managers and executives to quickly understand an employee's situation and make informed retention decisions.

### 🎯 Profile Header Section

#### Employee Identification
- **Large Employee Name** (2.5rem font, bold, white)
  - Prominently displayed at the top
  - Easy to read and identify
- **Employee ID** (small text, gray)
  - Displayed below name
  - Format: "ID: EMP001"

#### Quick Info Line
- **Department** (bold, if available)
- **Job Role** (bold, if available)
- **Manager Name** (if available)
- Format: "**Department** • **Role** • Manager: **Name**"

#### Risk Badge (Right Side)
A prominent visual indicator showing:
- **Risk Level Label**: "Risk Level" (small uppercase text)
- **Risk Score**: Large percentage (32px font, color-coded)
- **Risk Level Name**: Text label (High/Critical/etc.)
- **Progress Bar**: Visual representation of risk percentage
- **Color Coding**:
  - 🟢 Low: Green (#34C759)
  - 🟡 Moderate: Yellow (#FFCC00)
  - 🟠 High: Orange (#FF9500)
  - 🔴 Critical: Red (#FF3B30)

### 📋 Additional Information Section (Top)

**Purpose**: Shows any data fields not covered in main sections, excluding JSON fields.

**Display Logic**:
- Only shows fields that have actual data (not empty, not "N/A", not zero)
- Excludes all JSON fields (`*_JSON`)
- Displays in responsive grid (up to 3 columns)
- Each field shown as a styled data card

**Format**: Field name (uppercase, small) → Value (large, readable)

### 📊 Detailed Data Grid

The profile organizes information into **6 main sections**:

#### 1. 👤 Personal & Education
**Fields Displayed** (only if data exists):
- Age
- Gender
- Education Level
- Education Field
- Marital Status

**Layout**: Responsive grid (up to 5 columns)

#### 2. 💼 Employment Details
**Fields Displayed**:
- Job Level
- Over Time (Yes/No)
- Business Travel Frequency
- Distance from Home (km)
- Number of Companies Worked

**Layout**: Responsive grid (up to 5 columns)

#### 3. 💰 Compensation
**Fields Displayed**:
- Monthly Income (formatted as currency: $X,XXX)
- Percent Salary Hike (formatted as percentage: X%)
- Stock Option Level (allows zero as meaningful)
- Performance Rating

**Layout**: Responsive grid (up to 4 columns)

#### 4. ⏳ History & Tenure
**Fields Displayed** (allows zero as meaningful):
- Total Working Years
- Years at Company
- Years in Current Role
- Years with Current Manager
- Training Times Last Year

**Layout**: Responsive grid (up to 5 columns)

#### 5. 😊 Satisfaction (1-4 Scale)
**Fields Displayed**:
- Environment Satisfaction (1-4)
- Job Satisfaction (1-4)
- Work Life Balance (1-4)
- Relationship Satisfaction (1-4)

**Layout**: Responsive grid (up to 4 columns)

### 💬 Qualitative Feedback Section

**Purpose**: Highlights important narrative data that provides context beyond numbers.

#### Manager Notes
- **Format**: Styled box with blue left border
- **Background**: Dark gradient card
- **Layout**: Full-width, scrollable if long
- **Styling**: 
  - Label: Small uppercase text (gray)
  - Content: Readable text (white, line-height 1.6)
  - Preserves line breaks and formatting

#### Survey Comments
- **Format**: Same styling as Manager Notes
- **Purpose**: Employee's own words about their experience
- **Display**: Only if data exists

### 📈 Historical Data & Trends

#### Daily Work Hours Trend (Last 30 Days)
- **Chart Type**: Line chart (Plotly)
- **X-Axis**: Day (1-30)
- **Y-Axis**: Hours Worked
- **Color**: Blue (#007AFF)
- **Purpose**: Shows work pattern consistency, potential burnout indicators
- **Display**: Only if `Daily_Log_JSON` has data

#### Performance History (Quarterly)
- **Chart Type**: Line chart
- **X-Axis**: Quarter (Q1 Y1, Q2 Y1, ... Q4 Y3)
- **Y-Axis**: Performance Score
- **Color**: Green (#34C759)
- **Timeframe**: 3 years (12 quarters)
- **Purpose**: Shows performance trajectory over time
- **Display**: Only if `Performance_History_JSON` has data

#### Engagement History (Quarterly)
- **Chart Type**: Line chart
- **X-Axis**: Quarter
- **Y-Axis**: Engagement Score
- **Color**: Orange (#FF9500)
- **Timeframe**: 3 years (12 quarters)
- **Purpose**: Shows engagement trends, identifies decline patterns
- **Display**: Only if `Engagement_History_JSON` has data

### ⚠️ Top Risk Drivers Section

**Purpose**: Explains WHY the employee is at risk, with actionable insights.

#### Display Format:
- **Expandable Cards**: Each driver in its own expandable section
- **Ranking**: Numbered 1-3 (🔴 emoji indicator)
- **Driver Name**: Bold, with influence percentage
- **Description**: What the driver means for this employee
- **Benchmark**: Industry/peer comparison context
- **Progress Bar**: Visual representation of influence percentage

#### Example:
```
🔴 1. No Career Progression (45% influence)
   Description: No promotion in 24 months
   Benchmark: Peer benchmarking shows 3.2x higher departure probability
   [Progress Bar: 45%]
```

### 📝 JSON Data Expanders

**Purpose**: Provides access to raw JSON data for detailed analysis.

**Sections** (3 columns):
1. **📝 Daily Log (JSON)**: Expandable JSON viewer
2. **📈 Performance History (JSON)**: Expandable JSON viewer
3. **💚 Engagement History (JSON)**: Expandable JSON viewer

**Display**: Only if data exists, otherwise shows "No data available" message

### 🎨 Design Philosophy

#### Visual Hierarchy
1. **Name & Risk Badge**: Most prominent (top)
2. **Additional Info**: Quick context (if available)
3. **Detailed Sections**: Organized by category
4. **Qualitative Feedback**: Highlighted with special styling
5. **Historical Trends**: Visual charts for pattern recognition
6. **Risk Drivers**: Actionable insights
7. **Raw Data**: Available but not prominent

#### Data Display Rules
- **No Empty Fields**: Only shows fields with actual data
- **No "N/A" Values**: Filters out placeholder text
- **Smart Formatting**: Currency, percentages, distances formatted appropriately
- **Zero Handling**: Some fields (Stock Options, Tenure) allow zero as meaningful
- **Responsive Layout**: Adapts to screen size (1-5 columns)

#### Color Coding
- **Risk Levels**: Green → Yellow → Orange → Red
- **Charts**: Blue (work hours), Green (performance), Orange (engagement)
- **Background**: Dark theme (#0E1117) for professional appearance
- **Cards**: Gradient backgrounds with subtle borders

### 🔍 Why This Design?

1. **Quick Scanning**: Large name and risk badge allow instant identification
2. **Progressive Disclosure**: Most important info first, details below
3. **Visual Patterns**: Charts make trends immediately obvious
4. **Actionable Insights**: Risk drivers explain what to do next
5. **Complete Picture**: Combines quantitative and qualitative data
6. **Professional Aesthetic**: Dark theme, clean layout, Apple-inspired design

---

## 📚 Complete App Documentation

### 🏗️ Application Architecture

#### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend Framework** | Streamlit | Python-based web app framework |
| **Data Processing** | Pandas, NumPy | DataFrame operations, statistical analysis |
| **Visualization** | Plotly Express, Plotly Graph Objects | Interactive charts and graphs |
| **ML/AI** | Custom Rule-Based System + Groq API (Llama 3.3 70B) | Risk prediction and natural language processing |
| **Styling** | CSS (embedded) | Dark theme, Apple-inspired UI |
| **Interactivity** | JavaScript (embedded) | Dynamic layout, auto-scrolling, tab detection |
| **Search** | RapidFuzz | Fuzzy employee name/ID matching |

#### File Structure

```
AI Class Project/
├── app.py                 # Main Streamlit application
├── predict.py             # Risk prediction engine
├── chatbot.py             # Rule-based chatbot
├── llm_chatbot.py         # LLM-powered chatbot (Groq)
├── employee_attrition_dataset_final.csv  # Employee data
├── logo.png               # Company logo
└── requirements.txt       # Python dependencies
```

### 🎯 Application Tabs

The app is organized into **4 main tabs**:

#### 1. 🏠 Dashboard Tab

**Purpose**: Executive overview of organizational risk

**Components**:

##### KPI Cards (Top Row)
- **Total Employees**: Count of all employees
- **High Risk**: Count of employees at 60%+ risk
- **Critical**: Count of employees at 85%+ risk
- **Est. Attrition Cost**: Calculated financial risk (see cost calculation above)

##### Chart 1: Global Risk Distribution
- **Type**: Donut/Pie Chart
- **Shows**: Percentage of employees in each risk category
- **Colors**: Green (Low), Yellow (Moderate), Orange (High), Red (Critical)
- **Purpose**: Quick visual of overall risk landscape

##### Chart 2: Average Risk by Department
- **Type**: Bar Chart
- **Shows**: Average risk score per department
- **Purpose**: Identify departments needing attention

##### Chart 3: Risk Composition by Department
- **Type**: Stacked Bar Chart
- **Shows**: Count of Low/Moderate/High/Critical per department
- **Purpose**: Understand risk distribution within departments

##### Chart 4: Attrition Risk Curve (Tenure)
- **Type**: Line Chart
- **Shows**: Average risk score vs. years of tenure
- **Purpose**: Identify if risk is higher for new hires or veterans
- **Filter**: Only shows tenure < 15 years (removes outliers)

##### Chart 5: Risk by Job Level
- **Type**: Horizontal Bar Chart
- **Shows**: Average risk score by job level/seniority
- **Color Scale**: Blue-Red gradient
- **Purpose**: Understand if risk varies by seniority

##### Chart 6: Top Retention Priorities
- **Type**: Data Table
- **Shows**: Top 5 high/critical risk employees
- **Columns**: Name, Department, Risk Score, Risk Level
- **Purpose**: Actionable list for immediate intervention

#### 2. 🔍 Search & Profile Tab

**Purpose**: Find and view detailed employee profiles

**Components**:

##### Search Interface
- **Text Search**: Search by name or employee ID (fuzzy matching)
- **Department Filter**: Multi-select dropdown
- **Risk Level Filter**: Multi-select (Low/Moderate/High/Critical)
- **Results**: Shows matching employees with risk scores

##### Employee Profile View
(Detailed in [Individual Employee Profile](#individual-employee-profile) section above)

**Key Features**:
- **Back Button**: Returns to search view
- **Dynamic Field Display**: Only shows fields with data
- **Visual Risk Indicator**: Prominent risk badge
- **Historical Charts**: Performance, engagement, work hours trends
- **Risk Drivers**: Top 3 with explanations
- **Qualitative Feedback**: Manager notes and survey comments

#### 3. 📈 Department Analysis Tab

**Purpose**: Deep dive into department-level risk patterns

**Components**:

##### Department Selector
- **Dropdown**: Select department to analyze
- **Options**: All departments from dataset

##### Metrics Cards
- **Total Employees**: Count in selected department
- **Avg Risk Score**: Average risk percentage
- **High Risk Count**: Number of employees at 60%+ risk

##### Risk vs. Tenure Scatter Plot
- **Type**: Scatter Chart
- **X-Axis**: Tenure (years)
- **Y-Axis**: Risk Score (%)
- **Color**: Risk Level
- **Hover**: Shows employee name
- **Purpose**: Identify if risk correlates with tenure in this department

##### Employee List Table
- **Columns**: Employee Name, Risk Score, Risk Level
- **Sort**: By risk score (highest first)
- **Format**: Risk score as percentage
- **Purpose**: See all employees in department ranked by risk

#### 4. 💬 AI Assistant Tab

**Purpose**: Natural language interface for HR analytics

**Components**:

##### Chat Interface
- **Message Container**: Scrollable area for conversation
- **Input Bar**: Fixed at bottom, pill-shaped design
- **Welcome Message**: Shown only when no messages exist
- **Loading Indicator**: Animated dots while processing

##### AI Capabilities

The AI Assistant can:

1. **Answer Questions About Risk**
   - "How many employees are at high risk?"
   - "Which department has the highest risk?"
   - "Show me risk statistics"

2. **Analyze Departments**
   - "Engineering department analysis"
   - "Compare Sales and Marketing"
   - "Which department needs the most attention?"

3. **Find Employees**
   - "Find John Smith"
   - "Search EMP001"
   - "Who is at critical risk?"

4. **Generate Visualizations**
   - "Create a chart comparing departments"
   - "Show me risk distribution"
   - "Visualize high risk employees"

5. **Provide Recommendations**
   - "What should we prioritize?"
   - "Give me retention recommendations"
   - "What actions should we take?"

6. **Pattern Analysis**
   - "What patterns do you see?"
   - "Analyze trends in the data"
   - "What insights can you provide?"

##### AI Implementation

**Two Modes Available**:

1. **LLM Mode** (Primary - if API key available):
   - Uses **Groq API** with **Llama 3.3 70B Versatile** model
   - Accesses real-time employee data context
   - Provides intelligent, contextual responses
   - Supports streaming responses for better UX

2. **Rule-Based Mode** (Fallback):
   - Uses keyword matching and predefined responses
   - Provides structured answers based on query patterns
   - Works without API key

**Data Context Injection**:
- Risk summary statistics
- Department analysis
- Top high-risk employees
- Risk distribution breakdown

**Conversation Features**:
- **Follow-up Questions**: Understands context from previous messages
- **Conversational**: Handles greetings, thanks, acknowledgments
- **Streaming**: Responses appear word-by-word for better UX
- **Chart Generation**: Can create and display interactive charts

### 🎨 UI/UX Design

#### Design Theme
- **Style**: Minimalist, Apple-inspired dark theme
- **Color Scheme**: 
  - Background: #0E1117 (very dark blue-gray)
  - Cards: #1E293B (dark slate)
  - Text: #FFFFFF (white)
  - Accents: #60A5FA (blue), #007AFF (iOS blue)
- **Font**: Inter (Google Fonts) - clean, modern, professional

#### Navigation
- **Tabs**: Sticky navigation ribbon at top
- **Logo**: Positioned on right side of navigation (50px height)
- **Tab Styling**: 
  - Inactive: Dark gray background
  - Active: Blue background with shadow
  - Hover: Light gray background

#### Responsive Design
- **Grid Layouts**: Adapts from 1-5 columns based on screen size
- **Fixed Elements**: Input bar (chatbot) stays at bottom
- **Scrollable Areas**: Only message container scrolls, not entire page
- **Mobile-Friendly**: Touch-friendly button sizes, readable text

#### Visual Elements

##### Data Cards
- **Style**: Gradient background, subtle border
- **Hover Effect**: Slight lift and border highlight
- **Layout**: Responsive grid
- **Content**: Label (small, uppercase) + Value (large, bold)

##### Charts
- **Theme**: Dark background, white text
- **Interactivity**: Hover tooltips, zoom, pan
- **Colors**: Consistent color coding (risk levels, metrics)

##### Message Bubbles (Chatbot)
- **User Messages**: Dark gray, right-aligned, rounded
- **Assistant Messages**: Transparent, left-aligned, with icon
- **Timestamps**: Small gray text below messages

### 🔧 Key Functions & Methods

#### Prediction Engine (`predict.py`)

| Function | Purpose | Returns |
|---------|---------|---------|
| `predict_attrition_risk(employee_row)` | Main prediction function | Dictionary with risk score, level, drivers |
| `extract_employee_features(employee_row)` | Extract numerical features | Dictionary of feature values |
| `calculate_risk_score(features)` | Calculate 0-100% risk score | Float (0-100) |
| `get_risk_level(score)` | Determine risk category | Tuple (level, emoji) |
| `calculate_top_risk_drivers(features, n=3)` | Identify top risk factors | List of driver dictionaries |
| `get_risk_summary(df)` | Organization-wide statistics | Dictionary with counts and averages |
| `get_department_summary(df)` | Department-level analysis | Dictionary keyed by department |
| `get_high_risk_employees(df, min_level)` | Filter high-risk employees | List of employee predictions |

#### Chatbot (`chatbot.py`)

| Function | Purpose | Returns |
|---------|---------|---------|
| `process_query(query, ui_context, conversation_history)` | Process user query | Response dictionary |
| `search_employees(search_term, limit, min_score)` | Fuzzy employee search | List of matching employees |
| `_handle_risk_overview(query, context)` | Answer risk questions | Formatted response |
| `_handle_department_analysis(query, context)` | Department queries | Department statistics |
| `_handle_chart_request(query, context)` | Generate charts | Chart object |
| `_handle_recommendations(query, context)` | Provide recommendations | Action items list |

#### LLM Chatbot (`llm_chatbot.py`)

| Function | Purpose | Returns |
|---------|---------|---------|
| `process_query(query, ui_context, conversation_history)` | LLM-powered query processing | Response dictionary |
| `process_query_stream(query, ui_context, conversation_history)` | Streaming LLM response | Generator of text chunks |
| `_build_data_context(force_rebuild)` | Build data summary for LLM | Formatted context string |
| `_load_api_key()` | Load API key from config | API key string or None |

#### Main App (`app.py`)

| Function | Purpose | Returns |
|---------|---------|---------|
| `load_data()` | Load CSV dataset | DataFrame |
| `prepare_dashboard_data(df)` | Process data for charts | Processed DataFrame |
| `safe_parse_json(json_str)` | Safely parse JSON | List or dictionary |
| `has_data(value, allow_zero)` | Check if value has data | Boolean |
| `format_value(value, field_type)` | Format value for display | Formatted string |

### 📊 Data Flow

```
CSV File
   ↓
load_data() → DataFrame
   ↓
For each employee:
   ↓
extract_employee_features() → Feature Dictionary
   ↓
calculate_risk_score() → Risk Score (0-100%)
   ↓
get_risk_level() → Risk Level (Low/Moderate/High/Critical)
   ↓
calculate_top_risk_drivers() → Top 3 Drivers List
   ↓
predict_attrition_risk() → Complete Prediction Dictionary
   ↓
Display in UI (Dashboard/Profile/Search)
```

### 🔐 Configuration

#### API Keys
- **Location**: `config.env` (in the AI Class Project directory)
- **Required**: `GROQ_API_KEY=your_key_here`
- **Fallback**: If not available, chatbot will display a placeholder message

#### Data File
- **Path**: `employee_attrition_dataset_final.csv`
- **Required Columns**: See [Inputs](#-inputs) section
- **Format**: CSV with JSON columns for historical data

### 🚀 Running the Application

#### Prerequisites
```bash
pip install -r requirements.txt
```

#### Start Application
```bash
streamlit run app.py
```

#### Access
- **URL**: `http://localhost:8501`
- **Browser**: Any modern browser (Chrome, Firefox, Safari, Edge)

### 📈 Performance Optimizations

1. **Caching**: 
   - `@st.cache_data` on data loading functions
   - Context caching in LLM chatbot
   - DataFrame hash checking for cache invalidation

2. **Lazy Processing**:
   - Only processes employees when needed
   - Limits batch processing to 200-5000 rows
   - Samples large datasets for dashboard charts

3. **Streaming Responses**:
   - LLM responses stream word-by-word
   - Improves perceived latency
   - Better user experience

4. **Efficient Search**:
   - Fuzzy matching with score thresholds
   - Limits results to top matches
   - Caches search index

### 🐛 Error Handling

- **Missing Data**: Gracefully handles missing fields, shows "N/A" or hides empty fields
- **JSON Parsing**: Safe parsing with fallback to empty list/dict
- **API Failures**: Falls back to rule-based chatbot if LLM unavailable
- **File Not Found**: Shows error message, continues with empty dataset
- **Invalid Input**: Validates user input, provides helpful error messages

### 🔄 Future Enhancements

Potential improvements:
- Export functionality (PDF reports, CSV exports)
- Email alerts for critical risk employees
- Integration with HRIS systems
- Advanced ML models (XGBoost, Random Forest)
- Real-time data updates
- Multi-language support
- Mobile app version

---

## 📝 Summary

The Employee Attrition Risk Prediction System is a **comprehensive HR analytics platform** that:

✅ **Predicts** employee departure risk using 10 weighted risk drivers  
✅ **Visualizes** organizational risk through interactive charts  
✅ **Analyzes** department-level patterns and trends  
✅ **Provides** actionable recommendations for retention  
✅ **Answers** questions through AI-powered natural language interface  
✅ **Calculates** estimated financial impact of potential attrition  
✅ **Displays** detailed employee profiles with historical trends  

The system combines **data science**, **machine learning**, and **user experience design** to help HR teams make data-driven decisions about employee retention.

---

**Version**: 2.0  
**Last Updated**: 2024  
**Technology**: Python, Streamlit, Plotly, Groq API (Llama 3.3 70B)



# Quick Start Guide

This guide will help you get the Employee Attrition Risk Dashboard up and running quickly.

## Prerequisites

- Python 3.8 or higher installed
- pip (usually comes with Python)
- Terminal/Command Prompt access

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hr-risk-analysis.git
cd hr-risk-analysis
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages (pandas, numpy, scikit-learn, streamlit, plotly, etc.)

### 4. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser. If it doesn't, look for a URL in the terminal (usually `http://localhost:8501`).

## What You'll See

The dashboard has three main sections:

1. **Dashboard** - Overview with key metrics and charts
2. **Employee Profile** - Detailed analysis of individual employees
3. **Department Analysis** - Department-level risk insights

## Common Issues & Solutions

### "Command not found: python3"
- Try `python` instead of `python3`
- On Windows, use `python` or `py`

### "pip: command not found"
- Try `python -m pip` instead of `pip`
- On Windows, use `python -m pip`

### "Port 8501 is already in use"
- Streamlit will automatically use the next available port
- Check the terminal output for the correct URL

### "ModuleNotFoundError"
- Make sure your virtual environment is activated (you should see `(venv)` in your prompt)
- Run `pip install -r requirements.txt` again

### Dashboard won't load
- Check that all files are present (especially `model/` folder and `employee_attrition_dataset_final.csv`)
- Ensure you're in the project directory when running `streamlit run app.py`

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the Streamlit server.

## Next Steps

- Explore the dashboard features
- Check individual employee profiles
- Analyze department-level risks
- Read `README.md` for more detailed information

## Need Help?

- Check `README.md` for comprehensive documentation
- Review error messages in the terminal
- Ensure all prerequisites are met

---

**That's it!** You should now have the dashboard running. Enjoy exploring the Employee Attrition Risk Prediction System!


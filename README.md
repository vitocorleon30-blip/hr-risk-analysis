# Employee Attrition Risk Prediction System

Enterprise-grade AI-driven predictive model for identifying employees at risk of voluntary departure, enabling proactive retention interventions. Features a professional F500 corporate dashboard built with Streamlit.

## 🎯 Overview

This system provides:
- **4-Tier Risk Scoring**: Low (0-30%), Moderate (30-60%), High (60-85%), Critical (85-100%)
- **Top 3 Risk Drivers**: Identifies most impactful factors with influence percentages
- **Risk Trend Analysis**: Shows if risk is increasing (↑), stable (→), or decreasing (↓)
- **Comprehensive Data**: 3 years of historical records, performance reviews, engagement surveys
- **Explainable AI**: Natural language explanations for each prediction
- **Professional Dashboard**: F500 corporate-style web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hr-risk-analysis.git
   cd hr-risk-analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, sklearn, streamlit; print('✓ All dependencies installed')"
   ```

### Running the Dashboard

**Start the Streamlit dashboard:**
```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

If it doesn't open automatically, navigate to the URL shown in the terminal.

**Features of the Dashboard:**
- **Executive Dashboard**: Overview with KPIs, risk distribution, and department analysis
- **Employee Profile**: Detailed individual risk analysis with performance trends
- **Department Analysis**: Department-level risk insights and comparisons

## 📁 Project Structure

```
hr-risk-analysis/
├── README.md                              # This file
├── SETUP.md                               # Quick-start guide
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
├── FRONTEND_INTEGRATION_GUIDE.md          # Guide for frontend developers
├── employee_attrition_dataset_final.csv   # Main dataset (500 employees)
├── app.py                                 # ⭐ Streamlit dashboard application
├── predict.py                             # ⭐ Main prediction module
├── generate_dataset.py                    # Data generator script
├── train_model.py                         # Model training script
└── model/                                 # Trained model files
    ├── attrition_model.pkl                # Trained Random Forest model
    ├── scaler.pkl                         # Feature scaler
    └── feature_info.pkl                   # Feature metadata
```

## 🔧 Usage

### Using the Dashboard (Recommended)

Simply run `streamlit run app.py` and use the web interface to:
- View overall risk statistics
- Analyze individual employee profiles
- Compare departments
- Filter high-risk employees

### Using the Python API

```python
import pandas as pd
from predict import predict_attrition_risk, get_high_risk_employees

# Load data
df = pd.read_csv('employee_attrition_dataset_final.csv')

# Predict for one employee
prediction = predict_attrition_risk(df.iloc[0])
print(f"Risk Score: {prediction['risk_score']}%")
print(f"Risk Level: {prediction['risk_level']} {prediction['risk_color']}")

# Get all high-risk employees
high_risk = get_high_risk_employees(df, min_level='High')
for emp in high_risk:
    print(f"{emp['employee_name']}: {emp['risk_score']}%")
```

### Regenerate Dataset (Optional)

If you need to regenerate the employee dataset:
```bash
python generate_dataset.py
```

This creates `employee_attrition_dataset_final.csv` with 500 synthetic employees.

### Retrain Model (Optional)

If you need to retrain the model:
```bash
python train_model.py
```

This saves the trained model to `model/` folder.

## 🎯 Risk Drivers

The system monitors 10 key risk factors:

1. No Career Progression (18% weight)
2. Below-Market Compensation (16% weight)
3. Declining Manager Engagement (14% weight)
4. Declining Office Presence (12% weight)
5. Increasing Sick Leave (11% weight)
6. Declining Engagement Score (10% weight)
7. Training & Development Gap (6% weight)
8. Recognition Gap (5% weight)
9. Team Turnover Contagion (4% weight)
10. Communication Slowdown (4% weight)

## 🔬 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 25+ engineered features from employee data
- **Training Data**: 500 employees (90% normal, 10% at-risk)
- **Performance**: ~85% accuracy, ~88% recall on test set

## 📖 Documentation

- **Quick Start**: See `SETUP.md` for a simplified setup guide
- **Frontend Developers**: See `FRONTEND_INTEGRATION_GUIDE.md` for API documentation and integration examples
- **Data Scientists**: See code comments in `predict.py` and `train_model.py` for model details

## 🐛 Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError`**
- **Solution**: Make sure you've activated your virtual environment and installed all dependencies with `pip install -r requirements.txt`

**Issue: Dashboard won't start**
- **Solution**: Ensure Streamlit is installed: `pip install streamlit`. Check that port 8501 is not in use.

**Issue: Model files not found**
- **Solution**: The model files should be in the `model/` directory. If missing, run `python train_model.py` to generate them.

**Issue: Dataset file not found**
- **Solution**: Ensure `employee_attrition_dataset_final.csv` is in the project root directory. If missing, run `python generate_dataset.py`.

**Issue: Port already in use**
- **Solution**: Streamlit will try to use port 8501. If it's occupied, Streamlit will automatically try the next available port. Check the terminal output for the correct URL.

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify Python version is 3.8 or higher: `python --version`
3. Ensure you're in the project directory when running commands
4. Check the terminal output for specific error messages

## ⚠️ Important Notes

- The dataset contains synthetic data for demonstration purposes
- Model files (`.pkl`) must be present in `model/` folder for predictions
- For production use, retrain with real organizational data
- The dashboard requires an active internet connection for loading Google Fonts (Inter)

## 📝 License

This project is for educational/academic purposes.

## 👥 Credits

- **Backend**: Data generation, model training, prediction engine
- **Frontend**: F500 corporate dashboard design and implementation
- **Documentation**: Comprehensive guides and setup instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built for AI Class Project - Predictive Attrition Risk Modelling**

For questions or issues, refer to the documentation files or check the code comments for implementation details.

"""
Employee Attrition Risk Model Training
======================================
Trains a Random Forest classifier on the generated employee dataset
to predict attrition risk, with explainability features.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from typing import Dict, List, Tuple, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "employee_attrition_dataset_final.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, "feature_info.pkl")

RANDOM_STATE = 42

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical features from the raw dataset.
    
    Features:
    - Avg_Daily_Hours: Mean of daily work hours (from JSON)
    - Hours_Variance: Std deviation of daily hours
    - Performance_Current: Q4 performance score
    - Performance_Trend: Slope of Q1→Q4 performance
    - Engagement_Current: Q4 engagement score
    - Engagement_Trend: Slope of Q1→Q4 engagement
    - Sick_Days_60D: Direct
    - Vacation_Days_12M: Direct
    - Tenure_Years: Direct
    - Months_Since_Promotion: Direct
    """
    features = []
    
    for idx, row in df.iterrows():
        # Parse JSON columns
        daily_hours = json.loads(row['Daily_Log_JSON'])
        performance_history = json.loads(row['Performance_History_JSON'])
        engagement_history = json.loads(row['Engagement_History_JSON'])
        
        # Filter out weekends (zero hours) for accurate calculation
        working_hours = [h for h in daily_hours if h > 0]
        
        # Calculate features
        feature_dict = {
            'Employee_ID': row['Employee_ID'],
            
            # Work hours metrics
            'Avg_Daily_Hours': np.mean(working_hours) if working_hours else 0,
            'Hours_Variance': np.std(working_hours) if working_hours else 0,
            'Max_Daily_Hours': max(working_hours) if working_hours else 0,
            
            # Performance metrics
            'Performance_Current': performance_history[-1],  # Q4
            'Performance_Trend': calculate_trend(performance_history),
            'Performance_Avg': np.mean(performance_history),
            
            # Engagement metrics
            'Engagement_Current': engagement_history[-1],  # Q4
            'Engagement_Trend': calculate_trend(engagement_history),
            'Engagement_Avg': np.mean(engagement_history),
            
            # Direct metrics
            'Sick_Days_60D': row['Sick_Days_60D'],
            'Vacation_Days_12M': row['Vacation_Days_12M'],
            'Tenure_Years': row['Tenure_Years'],
            'Months_Since_Promotion': row['Months_Since_Promotion'],
        }
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def calculate_trend(values: List[int]) -> float:
    """
    Calculate the trend (slope) of a sequence of values.
    Positive = improving, Negative = declining, Near zero = stable
    """
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with interpretable parameters.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,           # Keep shallow for explainability
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle imbalanced classes (10% at-risk)
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RandomForestClassifier, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray,
                   feature_names: List[str]) -> Dict[str, Any]:
    """
    Evaluate model performance and extract feature importances.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=['Not At Risk', 'At Risk'])
    }
    
    # Feature importances
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))
    feature_importance = dict(sorted(feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True))
    
    metrics['feature_importance'] = feature_importance
    
    return metrics

# =============================================================================
# EXPLAINABILITY
# =============================================================================

def generate_risk_explanation(employee_data: Dict, 
                              feature_importances: Dict[str, float]) -> str:
    """
    Generate a natural language explanation for why an employee is at risk.
    Uses the feature values and their importances to craft the explanation.
    Infers risk patterns from data rather than explicit labels.
    """
    explanations = []
    
    # Check key risk factors
    if employee_data.get('Engagement_Trend', 0) < -1:
        explanations.append(f"declining engagement (dropped significantly over quarters)")
    
    if employee_data.get('Engagement_Current', 10) <= 3:
        explanations.append(f"very low current engagement score ({employee_data['Engagement_Current']}/10)")
    
    if employee_data.get('Avg_Daily_Hours', 8) > 9.5:
        explanations.append(f"extreme work hours ({employee_data['Avg_Daily_Hours']:.1f}h average)")
    
    if employee_data.get('Vacation_Days_12M', 10) <= 2:
        explanations.append(f"minimal vacation taken ({employee_data['Vacation_Days_12M']} days in 12 months)")
    
    if employee_data.get('Sick_Days_60D', 0) >= 5:
        explanations.append(f"recent health concerns ({employee_data['Sick_Days_60D']} sick days in 60 days)")
    
    if employee_data.get('Months_Since_Promotion', 0) >= 36:
        explanations.append(f"stalled career progression ({employee_data['Months_Since_Promotion']} months since last promotion)")
    
    if employee_data.get('Tenure_Years', 0) >= 4 and employee_data.get('Months_Since_Promotion', 0) >= 36:
        explanations.append("long tenure without advancement")
    
    # Build explanation string
    if explanations:
        risk_factors = ", ".join(explanations)
        explanation = f"Risk indicators detected: {risk_factors}."
        
        # Infer risk pattern from data signals
        high_hours = employee_data.get('Avg_Daily_Hours', 8) > 9.5
        low_vacation = employee_data.get('Vacation_Days_12M', 10) <= 2
        high_sick = employee_data.get('Sick_Days_60D', 0) >= 5
        low_engagement = employee_data.get('Engagement_Current', 10) <= 4
        low_hours = employee_data.get('Avg_Daily_Hours', 8) < 7.5
        stalled = employee_data.get('Months_Since_Promotion', 0) >= 36
        
        if high_hours and (low_vacation or high_sick):
            explanation += " Profile suggests burnout risk - high performer under unsustainable workload."
        elif low_engagement and low_hours:
            explanation += " Profile suggests disengagement - employee may be doing minimum required work."
        elif stalled and employee_data.get('Tenure_Years', 0) >= 4:
            explanation += " Profile suggests frustration with career progression - retention risk."
    else:
        explanation = "Low risk profile - no significant warning signs detected."
    
    return explanation

def predict_with_explanation(model: RandomForestClassifier,
                             scaler: StandardScaler,
                             employee_features: pd.DataFrame,
                             feature_names: List[str],
                             feature_importances: Dict[str, float]) -> Dict[str, Any]:
    """
    Make a prediction with a natural language explanation.
    """
    # Scale features
    X = scaler.transform(employee_features[feature_names])
    
    # Predict
    risk_prob = model.predict_proba(X)[0, 1]
    is_at_risk = model.predict(X)[0]
    
    # Generate explanation
    employee_data = employee_features.iloc[0].to_dict()
    explanation = generate_risk_explanation(employee_data, feature_importances)
    
    # Determine risk level
    if risk_prob < 0.3:
        risk_level = "Low"
    elif risk_prob < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        'at_risk': bool(is_at_risk),
        'risk_probability': float(risk_prob),
        'risk_level': risk_level,
        'explanation': explanation,
        'top_risk_factors': list(feature_importances.keys())[:5]
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*60)
    print("EMPLOYEE ATTRITION RISK MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} employees")
    print(f"  At-Risk: {df['At_Risk'].sum()} ({df['At_Risk'].mean()*100:.1f}%)")
    
    # Feature engineering
    print("\n[2/6] Extracting features...")
    features_df = extract_features(df)
    
    # Define feature columns (exclude Employee_ID)
    feature_columns = [col for col in features_df.columns if col != 'Employee_ID']
    print(f"  Features: {feature_columns}")
    
    # Prepare data
    X = features_df[feature_columns].values
    y = df['At_Risk'].values
    
    # Scale features
    print("\n[3/6] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train model
    print("\n[5/6] Training Random Forest model...")
    model = train_model(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    print(f"  Cross-validation F1 scores: {cv_scores}")
    print(f"  Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Evaluate
    print("\n[6/6] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, feature_columns)
    
    print(f"\n  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")
    
    print("\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TN={cm[0][0]:3d}  FP={cm[0][1]:3d}")
    print(f"    FN={cm[1][0]:3d}  TP={cm[1][1]:3d}")
    
    print("\n  Feature Importances (Top 5):")
    for i, (feature, importance) in enumerate(list(metrics['feature_importance'].items())[:5]):
        print(f"    {i+1}. {feature}: {importance:.3f}")
    
    # Save model and artifacts
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {MODEL_PATH}")
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to: {SCALER_PATH}")
    
    # Save feature info
    feature_info = {
        'feature_columns': feature_columns,
        'feature_importances': metrics['feature_importance'],
        'metrics': {k: v for k, v in metrics.items() if k not in ['feature_importance', 'classification_report']}
    }
    with open(FEATURE_INFO_PATH, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"  Feature info saved to: {FEATURE_INFO_PATH}")
    
    # Demo prediction
    print("\n" + "="*60)
    print("DEMO: PREDICTING RISK FOR SAMPLE EMPLOYEES")
    print("="*60)
    
    # Find one at-risk and one normal employee
    at_risk_idx = df[df['At_Risk'] == 1].index[0]
    normal_idx = df[df['At_Risk'] == 0].index[0]
    
    for idx, label in [(at_risk_idx, "AT-RISK EMPLOYEE"), (normal_idx, "NORMAL EMPLOYEE")]:
        print(f"\n{label}:")
        employee = df.iloc[idx]
        employee_features = features_df[features_df['Employee_ID'] == employee['Employee_ID']]
        
        result = predict_with_explanation(
            model, scaler, employee_features, 
            feature_columns, metrics['feature_importance']
        )
        
        print(f"  Name: {employee['Name']}")
        print(f"  Department: {employee['Department']}")
        print(f"  Level: {employee['Level']}")
        print(f"  Predicted Risk: {result['risk_level']} ({result['risk_probability']*100:.1f}%)")
        print(f"  Explanation: {result['explanation']}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return model, scaler, feature_info

if __name__ == "__main__":
    model, scaler, feature_info = main()


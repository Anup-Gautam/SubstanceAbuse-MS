#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset - change this path to your actual cleaned data file location
df = pd.read_csv("Nsduh_2023.csv")

# Define fair features (no demographics except Health which is a strong predictor)
fair_features = [
    'AlcoholUse_PastMonth', 'MarijuanaUse_PastYear', 'CocaineUse_PastYear',
    'MethamphetamineUse_PastYear', 'OpioidMisuse_PastMonth', 'StimulantMisuse_PastMonth',
    'FentanylProductMisuse_Year', 'IllegallyMadeFentanyl_Use', 'Health'
]

# Make sure all features exist in the dataframe
fair_features = [f for f in fair_features if f in df.columns]

X = df[fair_features]
y = df['SubstanceUseDisorder']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class imbalance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)  # Using resampled data for better class balance

# Function to predict substance use disorder probability
def predict_substance_use_disorder(user_data):
    """
    Predicts the probability of substance use disorder based on user inputs.
    
    Parameters:
    user_data (dict): Dictionary containing user data for prediction
    
    Returns:
    tuple: (prediction (0/1), probability of substance use disorder)
    """
    # Create DataFrame from user data
    input_df = pd.DataFrame([user_data])
    
    # Check for missing columns and add them with default value
    missing_cols = [col for col in fair_features if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0
    
    # Ensure columns are in the correct order
    input_df = input_df[fair_features]
    
    # Make prediction
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1]  # Probability of class 1
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
    # Example: Person who uses multiple substances
    example_data = {
        'AlcoholUse_PastMonth': 1,     # Yes
        'MarijuanaUse_PastYear': 1,    # Yes
        'CocaineUse_PastYear': 1,      # Yes
        'MethamphetamineUse_PastYear': 0,  # No
        'OpioidMisuse_PastMonth': 1,   # Yes
        'StimulantMisuse_PastMonth': 1,  # Yes
        'FentanylProductMisuse_Year': 1,  # Yes
        'IllegallyMadeFentanyl_Use': 1,  # Yes
        'Health': 2  # Poor health (assuming lower values = better health)
    }
    
    pred, prob = predict_substance_use_disorder(example_data)
    print(f"Substance Use Disorder Prediction: {'Yes' if pred == 1 else 'No'}")
    print(f"Probability of Substance Use Disorder: {prob:.2%}")
    
    # Example: Person with only alcohol use
    low_risk_example = {
        'AlcoholUse_PastMonth': 1,     # Yes
        'MarijuanaUse_PastYear': 0,    # No
        'CocaineUse_PastYear': 0,      # No
        'MethamphetamineUse_PastYear': 0,  # No
        'OpioidMisuse_PastMonth': 0,   # No
        'StimulantMisuse_PastMonth': 0,  # No
        'FentanylProductMisuse_Year': 0,  # No
        'IllegallyMadeFentanyl_Use': 0,  # No
        'Health': 1  # Good health
    }
    
    pred, prob = predict_substance_use_disorder(low_risk_example)
    print(f"Substance Use Disorder Prediction: {'Yes' if pred == 1 else 'No'}")
    print(f"Probability of Substance Use Disorder: {prob:.2%}")
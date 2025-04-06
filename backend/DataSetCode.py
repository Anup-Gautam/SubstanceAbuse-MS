#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the drug consumption dataset with improved data handling.
    """
    # Load data
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    
    # Process drug usage columns
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 
                   'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
                   'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Convert drug codes to numerical values (CL0, CL1, etc.)
    for col in drug_columns:
        if col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].astype(str).str.replace('CL', '').astype(int)
    
    # One-hot encode categorical features
    categorical_cols = ['Gender', 'Education', 'Country', 'Ethnicity']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Create weighted features to better represent different drug risks
    # Defining drug risk weights based on potential harm (these weights should be calibrated by experts)
    drug_weights = {
        'Alcohol': 1.0,
        'Amphet': 1.5,
        'Amyl': 1.0,
        'Benzos': 1.5,
        'Cannabis': 0.8,
        'Coke': 2.0,
        'Crack': 2.5,
        'Ecstasy': 1.3,
        'Heroin': 3.0,
        'Ketamine': 1.5,
        'Legalh': 0.5,
        'LSD': 1.0,
        'Meth': 2.5,
        'Mushrooms': 0.8,
        'Nicotine': 0.7,
        'Semer': 0.7,
        'VSA': 1.2
    }
    
    # Calculate weighted drug usage
    df_encoded['Weighted_Drug_Usage'] = 0
    for drug in drug_columns:
        if drug in df_encoded.columns:
            # Apply exponential scaling to usage frequency to emphasize daily use
            # Usage scale: 0=never, 6=daily use
            df_encoded[f'{drug}_Weighted'] = df_encoded[drug] * drug_weights.get(drug, 1.0) * (1.2 ** df_encoded[drug])
            df_encoded['Weighted_Drug_Usage'] += df_encoded[f'{drug}_Weighted']
    
    # Count number of drugs used at all (any frequency above 0)
    df_encoded['Num_Drugs_Used'] = (df_encoded[drug_columns] > 0).sum(axis=1)
    
    # Count number of drugs used regularly (frequency of 3 or above - monthly+)
    df_encoded['Num_Regular_Drugs'] = (df_encoded[drug_columns] >= 3).sum(axis=1)
    
    # Count number of drugs used heavily (frequency of 5 or above - daily+)
    df_encoded['Num_Heavy_Drugs'] = (df_encoded[drug_columns] >= 5).sum(axis=1)
    
    # Calculate dependence risk based on frequency patterns
    # This formula emphasizes: 
    # 1. Number of drugs used heavily (daily)
    # 2. Weighted usage based on drug risk
    # 3. Diversity of drug use
    
    # Normalize the weighted usage (0-1 scale)
    max_weighted = df_encoded['Weighted_Drug_Usage'].max()
    df_encoded['Normalized_Weighted_Usage'] = df_encoded['Weighted_Drug_Usage'] / max_weighted
    
    # Calculate risk index with greater emphasis on heavy use
    df_encoded['Risk_Index'] = (
        0.4 * df_encoded['Normalized_Weighted_Usage'] + 
        0.3 * (df_encoded['Num_Heavy_Drugs'] / len(drug_columns)) +
        0.2 * (df_encoded['Num_Regular_Drugs'] / len(drug_columns)) +
        0.1 * (df_encoded['Num_Drugs_Used'] / len(drug_columns))
    )
    
    # Ensure the Risk_Index is between 0 and 1
    df_encoded['Risk_Index'] = df_encoded['Risk_Index'].clip(0, 1)
    
    # Create categorical risk levels for easy interpretation
    def categorize_risk(risk_index):
        if risk_index < 0.2:
            return 'very low risk'
        elif risk_index < 0.4:
            return 'low risk'
        elif risk_index < 0.6:
            return 'medium risk'
        elif risk_index < 0.8:
            return 'high risk'
        else:
            return 'very high risk'
    
    df_encoded['Risk_Level'] = df_encoded['Risk_Index'].apply(categorize_risk)
    
    # Analyze the distribution of the risk index to ensure it's reasonable
    print("\nRisk Index Distribution:")
    print(df_encoded['Risk_Index'].describe())
    
    # Create bins and count data points in each risk category
    risk_counts = df_encoded['Risk_Level'].value_counts().sort_index()
    print("\nRisk Level Distribution:")
    print(risk_counts)
    
    return df_encoded

def validate_risk_calculation(df):
    """
    Function to validate the risk calculation with example scenarios
    """
    print("\nValidating risk calculation with example cases:")
    
    # Example 1: Heavy single drug use
    print("\nExample 1: Heavy single drug use (daily alcohol only)")
    heavy_single = df[df['Alcohol'] == 6].copy()
    heavy_single = heavy_single[heavy_single[['Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 
                   'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
                   'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']].sum(axis=1) == 0]
    if not heavy_single.empty:
        print(f"  Risk Index range: {heavy_single['Risk_Index'].min():.2f} - {heavy_single['Risk_Index'].max():.2f}")
        print(f"  Most common risk level: {heavy_single['Risk_Level'].mode()[0]}")
    else:
        print("  No example found in dataset")
    
    # Example 2: Multiple heavy drug use
    print("\nExample 2: Heavy multiple drug use (3+ drugs used daily)")
    heavy_multi = df[df['Num_Heavy_Drugs'] >= 3].copy()
    if not heavy_multi.empty:
        print(f"  Risk Index range: {heavy_multi['Risk_Index'].min():.2f} - {heavy_multi['Risk_Index'].max():.2f}")
        print(f"  Most common risk level: {heavy_multi['Risk_Level'].mode()[0]}")
    else:
        print("  No example found in dataset")
    
    # Example 3: Occasional use of multiple drugs
    print("\nExample 3: Occasional use of multiple drugs (3+ drugs used in last year)")
    occasional = df[(df[['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 
                   'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
                   'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']] == 2).sum(axis=1) >= 3].copy()
    if not occasional.empty:
        print(f"  Risk Index range: {occasional['Risk_Index'].min():.2f} - {occasional['Risk_Index'].max():.2f}")
        print(f"  Most common risk level: {occasional['Risk_Level'].mode()[0]}")
    else:
        print("  No example found in dataset")
    
    return

def build_drug_abuse_model(df):
    """
    Build and train a regression model to predict drug abuse risk index (0-1)
    with improved hyperparameter tuning and cross-validation.
    """
    # Define features and target
    # Exclude the constructed risk metrics and drug weighted columns from features
    exclude_cols = ['Risk_Index', 'Risk_Level', 'Weighted_Drug_Usage', 
                    'Normalized_Weighted_Usage']
    exclude_cols.extend([col for col in df.columns if '_Weighted' in col])
    
    X = df.drop(exclude_cols, axis=1)
    y = df['Risk_Index']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Select only numerical columns for the model
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    X_train = X_train[numerical_cols]
    X_test = X_test[numerical_cols]
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Perform hyperparameter tuning with cross-validation
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [10, 15, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_pipeline = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model with cross-validation
    cv_scores = cross_val_score(
        best_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
    )
    print(f"Cross-validation MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    
    # Ensure predictions are within [0,1]
    y_pred = np.clip(y_pred, 0, 1)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test set Mean Squared Error: {mse:.4f}")
    print(f"Test set R² Score: {r2:.4f}")
    
    # Check for overfitting
    y_train_pred = best_pipeline.predict(X_train)
    y_train_pred = np.clip(y_train_pred, 0, 1)
    train_mse = mean_squared_error(y_train, y_train_pred)
    
    print(f"Training set MSE: {train_mse:.4f}")
    print(f"Difference (Train-Test): {train_mse - mse:.4f}")
    
    if train_mse < 0.5 * mse:
        print("Warning: Model may be overfitting. Consider regularization or simpler model.")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Risk Index')
    plt.ylabel('Predicted Risk Index')
    plt.title('Actual vs Predicted Risk Index')
    plt.savefig('risk_prediction_validation.png')
    print("Validation plot saved as 'risk_prediction_validation.png'")
    
    # Feature importance
    feature_importances = best_pipeline.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': numerical_cols,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance_df.head(10))
    
    # Save the model and preprocessing artifacts
    joblib.dump(best_pipeline, 'drug_abuse_index_model.pkl')
    joblib.dump(list(numerical_cols), 'model_features.pkl')
    
    print("\nModel and preprocessing artifacts saved.")
    
    return best_pipeline, numerical_cols

def predict_drug_abuse_risk(user_data, pipeline, features):
    """
    Predict drug abuse risk index (0-1) for a new user based on their data.
    
    Parameters:
    user_data: Dictionary with user features
    pipeline: Trained pipeline with scaler and model
    features: List of features used by the model
    
    Returns:
    Risk index (0-1) and associated risk level
    """
    # Create DataFrame from user data
    user_df = pd.DataFrame([user_data])
    
    # Calculate additional features needed by the model
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 
                   'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
                   'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Initialize missing drug columns with 0
    for drug in drug_columns:
        if drug not in user_df.columns:
            user_df[drug] = 0
    
    # Count number of drugs used
    user_df['Num_Drugs_Used'] = (user_df[drug_columns] > 0).sum(axis=1)
    user_df['Num_Regular_Drugs'] = (user_df[drug_columns] >= 3).sum(axis=1)
    user_df['Num_Heavy_Drugs'] = (user_df[drug_columns] >= 5).sum(axis=1)
    
    # Ensure all required features are present
    for feature in features:
        if feature not in user_df.columns:
            user_df[feature] = 0
    
    # Select only the features used by the model
    user_features = user_df[features]
    
    # Make prediction using the pipeline (which includes scaling)
    risk_index_raw = pipeline.predict(user_features)[0]
    
    # Ensure the index is within [0,1]
    risk_index = max(0, min(1, risk_index_raw))
    
    # Determine risk level based on index
    if risk_index < 0.2:
        risk_level = "very low risk"
    elif risk_index < 0.4:
        risk_level = "low risk"
    elif risk_index < 0.6:
        risk_level = "medium risk"
    elif risk_index < 0.8:
        risk_level = "high risk"
    else:
        risk_level = "very high risk"
    
    return {
        'risk_index': risk_index,
        'risk_level': risk_level
    }

def calculate_direct_risk(user_data):
    """
    Calculate risk directly with a more direct approach that ensures:
    1. If ALL drugs at level 0 (never used) -> risk index = 0
    2. If ALL drugs at level 6 (daily use) -> risk index = 1
    3. If even ONE drug at level 6 -> high risk (≥0.7)
    
    Parameters:
    user_data: Dictionary with user drug usage data
    
    Returns:
    Risk index (0-1) and associated risk level
    """
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 
                   'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
                   'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Defining drug risk weights based on potential harm
    drug_weights = {
        'Alcohol': 1.0,
        'Amphet': 1.5,
        'Amyl': 1.0,
        'Benzos': 1.5,
        'Cannabis': 0.8,
        'Coke': 2.0,
        'Crack': 2.5,
        'Ecstasy': 1.3,
        'Heroin': 3.0,
        'Ketamine': 1.5,
        'Legalh': 0.5,
        'LSD': 1.0,
        'Meth': 2.5,
        'Mushrooms': 0.8,
        'Nicotine': 0.7,
        'Semer': 0.7,
        'VSA': 1.2
    }
    
    # First check for edge cases:
    
    # 1. All drugs at 0 -> risk index = 0
    all_zero = True
    for drug in drug_columns:
        if user_data.get(drug, 0) > 0:
            all_zero = False
            break
    
    if all_zero:
        return {
            'risk_index': 0.0,
            'risk_level': "very low risk",
            'details': {
                'weighted_usage': 0,
                'max_single_drug_level': 0,
                'num_drugs_used': 0,
                'num_regular_drugs': 0,
                'num_heavy_drugs': 0
            }
        }
    
    # 2. All drugs at 6 -> risk index = 1
    all_six = True
    found_drugs = 0
    for drug in drug_columns:
        if drug in user_data:
            found_drugs += 1
            if user_data[drug] != 6:
                all_six = False
    
    if all_six and found_drugs > 0:
        return {
            'risk_index': 1.0,
            'risk_level': "very high risk",
            'details': {
                'weighted_usage': 100,  # Arbitrary high value
                'max_single_drug_level': 6,
                'num_drugs_used': found_drugs,
                'num_regular_drugs': found_drugs,
                'num_heavy_drugs': found_drugs
            }
        }
    
    # Initialize counters for normal case calculation
    weighted_usage = 0
    max_single_drug_level = 0
    max_weighted_single_drug = 0
    num_drugs_used = 0
    num_regular_drugs = 0
    num_heavy_drugs = 0
    has_daily_drug = False
    
    # Calculate weighted drug usage and counts
    for drug in drug_columns:
        usage = user_data.get(drug, 0)
        if usage > 0:
            num_drugs_used += 1
            
            # Track maximum usage level of any single drug
            if usage > max_single_drug_level:
                max_single_drug_level = usage
                
            # Apply exponential scaling to usage frequency with weight factor
            drug_weight = drug_weights.get(drug, 1.0)
            drug_score = usage * drug_weight * (1.5 ** usage)  # Increased exponential factor
            weighted_usage += drug_score
            
            # Track maximum weighted score of any single drug
            if drug_score > max_weighted_single_drug:
                max_weighted_single_drug = drug_score
            
            if usage >= 3:  # Used monthly or more
                num_regular_drugs += 1
                
            if usage >= 5:  # Used daily or more
                num_heavy_drugs += 1
                
            if usage == 6:  # Daily use
                has_daily_drug = True
    
    # Handle the case where at least one drug is at level 6 (daily use)
    # This should guarantee a high risk rating (at least 0.7)
    if has_daily_drug:
        # Base risk starts at 0.7 for a single daily drug
        base_risk = 0.7
        
        # Add additional risk based on number of drugs and their usage
        additional_risk = 0.3 * (weighted_usage / (max_weighted_single_drug * len(drug_columns)))
        
        # Combine for final risk score, ensuring we don't exceed 1.0
        risk_index = min(1.0, base_risk + additional_risk)
    else:
        # For non-daily use, calculate a more gradual risk scale
        
        # Maximum theoretical score if all drugs were at level 6 with max weights
        max_weight = max(drug_weights.values())
        max_possible_weighted = len(drug_columns) * 6 * max_weight * (1.5 ** 6)
        
        # Scale by number of drugs used and their intensity
        usage_factor = weighted_usage / max_possible_weighted
        
        # Calculate risk using a more aggressive formula for higher usage levels
        risk_index = usage_factor ** 0.7  # Using power less than 1 to raise risk for moderate usage
        
        # Boost risk based on maximum single drug level to ensure even moderate
        # use of a single drug registers as meaningful risk
        level_boost = (max_single_drug_level / 6) ** 2  # Quadratic scaling
        
        # Combine factors while ensuring we stay under the daily use threshold (0.7)
        risk_index = min(0.65, max(risk_index, level_boost))
    
    # Final risk index is guaranteed to be between 0 and 1
    risk_index = max(0, min(1, risk_index))
    
    # Determine risk level based on index
    if risk_index < 0.2:
        risk_level = "very low risk"
    elif risk_index < 0.4:
        risk_level = "low risk"
    elif risk_index < 0.6:
        risk_level = "medium risk"
    elif risk_index < 0.8:
        risk_level = "high risk"
    else:
        risk_level = "very high risk"
    
    return {
        'risk_index': risk_index,
        'risk_level': risk_level,
        'details': {
            'weighted_usage': weighted_usage,
            'max_single_drug_level': max_single_drug_level,
            'num_drugs_used': num_drugs_used,
            'num_regular_drugs': num_regular_drugs,
            'num_heavy_drugs': num_heavy_drugs,
            'has_daily_drug': has_daily_drug
        }
    }

def interpret_risk_index(risk_index):
    """
    Provide an interpretation of the risk index value.
    """
    if risk_index < 0.2:
        return "Very Low Risk: Minimal likelihood of problematic drug use. Even if using substances, usage patterns suggest responsible and infrequent consumption."
    elif risk_index < 0.4:
        return "Low Risk: Some drug use patterns present but generally low concern. Monitor for any increase in frequency or diversity of use."
    elif risk_index < 0.6:
        return "Moderate Risk: Notable drug use patterns that may warrant attention. Regular use of one or more substances indicates potential for developing problematic patterns."
    elif risk_index < 0.8:
        return "High Risk: Significant drug use patterns suggesting potential for dependency or abuse. Daily use of substances and/or use of multiple substances indicates concerning patterns."
    else:
        return "Very High Risk: Extensive drug use patterns indicating high probability of dependency or abuse. Frequent use of multiple substances, especially those with higher risk profiles, suggests intervention may be necessary."

def example_usage():
    """
    Example of how to use both the ML model and direct calculation for prediction.
    Shows examples of various edge cases and typical scenarios.
    """
    # Load the model and preprocessing artifacts
    try:
        pipeline = joblib.load('drug_abuse_index_model.pkl')
        features = joblib.load('model_features.pkl')
        model_available = True
    except:
        print("Model not found. Only direct calculation will be used.")
        model_available = False
    
    # Edge Case 1: All zeros - Should be 0 risk
    user_data_zero = {
        'Age': 2,
        'Alcohol': 0,
        'Cannabis': 0,
        'Nicotine': 0,
        'Amphet': 0,
        'Amyl': 0,
        'Benzos': 0,
        'Coke': 0,
        'Crack': 0,
        'Ecstasy': 0,
        'Heroin': 0,
        'Ketamine': 0,
        'Legalh': 0,
        'LSD': 0,
        'Meth': 0,
        'Mushrooms': 0,
        'Semer': 0,
        'VSA': 0
    }
    
    print("\nEdge Case 1: No drug use (all zeros)")
    direct_result = calculate_direct_risk(user_data_zero)
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level']}")
    
    # Edge Case 2: All sixes - Should be 1.0 risk
    user_data_all_six = {
        'Age': 2,
        'Alcohol': 6,
        'Cannabis': 6,
        'Nicotine': 6,
        'Amphet': 6,
        'Amyl': 6,
        'Benzos': 6,
        'Coke': 6,
        'Crack': 6,
        'Ecstasy': 6,
        'Heroin': 6,
        'Ketamine': 6,
        'Legalh': 6,
        'LSD': 6,
        'Meth': 6,
        'Mushrooms': 6,
        'Semer': 6,
        'VSA': 6
    }
    
    print("\nEdge Case 2: Maximum drug use (all sixes)")
    direct_result = calculate_direct_risk(user_data_all_six)
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level']}")
    
    # Example 1: Single drug at level 6 - Should be high risk (≥0.7)
    user_data_one_six = {
        'Age': 2,
        'Alcohol': 6,  # Only alcohol at level 6
        'Cannabis': 0,
        'Nicotine': 0,
        'Amphet': 0,
        'Amyl': 0,
        'Benzos': 0,
        'Coke': 0,
        'Crack': 0,
        'Ecstasy': 0,
        'Heroin': 0,
        'Ketamine': 0,
        'Legalh': 0,
        'LSD': 0,
        'Meth': 0,
        'Mushrooms': 0,
        'Semer': 0,
        'VSA': 0
    }
    
    print("\nExample 1: Single drug daily use (alcohol level 6)")
    direct_result = calculate_direct_risk(user_data_one_six)
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level']}")
    print(f"Interpretation: {interpret_risk_index(direct_result['risk_index'])}")
    
    # Example 2: High-risk drug at level 6 - Should be even higher risk
    user_data_heroin = {
        'Age': 2,
        'Alcohol': 0,
        'Cannabis': 0,
        'Nicotine': 0,
        'Amphet': 0,
        'Amyl': 0,
        'Benzos': 0,
        'Coke': 0,
        'Crack': 0,
        'Ecstasy': 0,
        'Heroin': 6,  # Only heroin at level 6
        'Ketamine': 0,
        'Legalh': 0,
        'LSD': 0,
        'Meth': 0,
        'Mushrooms': 0,
        'Semer': 0,
        'VSA': 0
    }
    
    print("\nExample 2: Single high-risk drug daily use (heroin level 6)")
    direct_result = calculate_direct_risk(user_data_heroin)
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level']}")
    print(f"Interpretation: {interpret_risk_index(direct_result['risk_index'])}")
    
    # Example 3: Moderate use of multiple drugs
    user_data_moderate = {
        'Age': 2,
        'Alcohol': 3,  # Monthly use
        'Cannabis': 3,  # Monthly use
        'Nicotine': 4,  # Weekly use
        'Amphet': 0,
        'Amyl': 0,
        'Benzos': 0,
        'Coke': 2,  # Yearly use
        'Crack': 0,
        'Ecstasy': 2,  # Yearly use
        'Heroin': 0,
        'Ketamine': 0,
        'Legalh': 0,
        'LSD': 0,
        'Meth': 0,
        'Mushrooms': 1,  # Used once
        'Semer': 0,
        'VSA': 0
    }
    
    print("\nExample 3: Moderate use of multiple drugs")
    direct_result = calculate_direct_risk(user_data_moderate)
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level']}")
    print(f"Interpretation: {interpret_risk_index(direct_result['risk_index'])}")
    
    # Example 4: Multiple heavy drug use
    user_data_heavy = {
        'Age': 2,
        'Alcohol': 6,  # Daily use
        'Cannabis': 6,  # Daily use
        'Nicotine': 6,  # Daily use
        'Amphet': 0,
        'Amyl': 0,
        'Benzos': 0,
        'Coke': 5,  # Used in last day
        'Crack': 0,
        'Ecstasy': 0,
        'Heroin': 0,
        'Ketamine': 0,
        'Legalh': 0,
        'LSD': 0,
        'Meth': 0,
        'Mushrooms': 0,
        'Semer': 0,
        'VSA': 0
    }
    
    print("\nExample 4: Multiple heavy drug use")
    direct_result = calculate_direct_risk(user_data_heavy)
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level']}")
    print(f"Interpretation: {interpret_risk_index(direct_result['risk_index'])}")
    
    # ML model prediction if available
    if model_available:
        print("\nML Model Predictions:")
        # Use one example to demonstrate ML model
        model_prediction = predict_drug_abuse_risk(user_data_heavy, pipeline, features)
        print(f"Risk Index (0-1): {model_prediction['risk_index']:.4f}")
        print(f"Risk Level: {model_prediction['risk_level']}")
        print("Note: ML model predictions may differ from direct calculations.")

def predict_for_new_user():
    """
    Interactive function to predict drug abuse risk for a new user.
    """
    # Load the model and preprocessing artifacts
    try:
        pipeline = joblib.load('drug_abuse_index_model.pkl')
        features = joblib.load('model_features.pkl')
        model_available = True
    except:
        print("Model not found. Only direct calculation will be used.")
        model_available = False
    
    print("\nEnter user information for drug abuse risk prediction:")
    
    # Get age category
    print("\nAge category:")
    print("1: 18-24 years")
    print("2: 25-34 years")
    print("3: 35-44 years")
    print("4: 45-54 years")
    print("5: 55+ years")
    age = int(input("Enter age category (1-5): "))
    
    # Get drug usage information
    # Scale: 0 (Never used) to 6 (Used daily)
    drug_data = {}
    drug_list = ['Alcohol', 'Cannabis', 'Nicotine', 'Cocaine', 'Ecstasy', 'Heroin', 'LSD', 
                'Amphet', 'Amyl', 'Benzos', 'Crack', 'Ketamine', 'Meth', 'Mushrooms']
    
    print("\nDrug usage scale:")
    print("0: Never used")
    print("1: Used only once")
    print("2: Used in last year")
    print("3: Used in last month")
    print("4: Used in last week")
    print("5: Used in last day")
    print("6: Used multiple times daily")
    
    for drug in drug_list:
        usage = int(input(f"Enter {drug} usage (0-6): "))
        drug_data[drug] = max(0, min(6, usage))  # Ensure value is between 0-6
    
    # Create full user data dictionary
    user_data = {'Age': age}
    user_data.update(drug_data)
    
    # Direct calculation (always available)
    direct_result = calculate_direct_risk(user_data)
    
    print("\n===== DRUG ABUSE RISK ASSESSMENT =====")
    print(f"Risk Index (0-1): {direct_result['risk_index']:.4f}")
    print(f"Risk Level: {direct_result['risk_level'].upper()}")
    print(f"Interpretation: {interpret_risk_index(direct_result['risk_index'])}")
    
    print("\nRisk Calculation Details:")
    

if __name__ == "__main__":
    # Main execution
    print("Improved Drug Abuse Risk Index Prediction Model (0-1 scale)")
    print("==========================================================")
    
    # Step 1: Load and preprocess data
    data_path = 'Drug_Consumption.csv'  # Update with your file path
    df = load_and_preprocess_data(data_path)
    
    if df is not None:
        # Validate the risk calculation
        validate_risk_calculation(df)
        
        # Step 2: Build and train the model (optional)
        pipeline, features = build_drug_abuse_model(df)
        
        # Step 3: Show example usage
        print("\nExample predictions:")
        example_usage()
        
        # Step 4: Interactive prediction
        choice = input("\nWould you like to enter data for a prediction? (y/n): ")
        if choice.lower() == 'y':
            predict_for_new_user()
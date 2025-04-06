#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime

class SubstanceAbuseRiskPredictor:
    """
    A comprehensive model for predicting substance abuse risk that combines:
    1. Direct risk calculation approach (from model 1)
    2. Machine learning risk factors (from model 2)
    3. Pharmacy visit frequency analysis
    4. Demographic risk factor weighting
    """
    
    def __init__(self):
        # Drug weights from model 1 - indicating potential harm
        self.drug_weights = {
            'Alcohol': 1.0,
            'Amphetamine': 1.5,
            'Amyl Nitrite': 1.0,
            'Benzodiazepines': 1.5,
            'Cannabis': 0.8,
            'Cocaine': 2.0,
            'Crack': 2.5,
            'Ecstasy': 1.3,
            'Heroin': 3.0,
            'Ketamine': 1.5,
            'Legal Highs': 0.5,
            'LSD': 1.0,
            'Methamphetamine': 2.5,
            'Mushrooms': 0.8,
            'Nicotine': 0.7,
            'Opioid Pain Relievers': 2.2,
            'Oxycodone': 2.3,
            'Hydrocodone': 2.2,
            'Fentanyl': 3.5,
            'Morphine': 2.0,
            'Methadone': 2.0,
            'Buprenorphine': 1.8,
            'Tramadol': 1.6,
            'Codeine': 1.5,
            'Stimulants': 1.7,
            'Adderall': 1.5,
            'Ritalin': 1.3,
            'Vyvanse': 1.4,
            'Sedatives': 1.5,
            'Xanax': 1.6,
            'Valium': 1.5,
            'Klonopin': 1.6,
            'Ativan': 1.5,
            'Ambien': 1.3,
            'Lunesta': 1.2,
            'Other': 1.0
        }
        
        # Usage scale mapping
        self.usage_scale = {
            "Never used": 0,
            "Used only once": 1,
            "Used in last year": 2,
            "Used in last month": 3,
            "Used in last week": 4,
            "Used in last day": 5,
            "Used multiple times daily": 6
        }
        
        # Define gender weights - reduced impact
        self.gender_weights = {
            "Male": 1.1,
            "Female": 1.0,
            "Other": 1.05
        }
        
        # Define race weights - reduced impact
        self.race_weights = {
            "Black": 1.08,
            "White": 1.05,
            "Asian": 1.02,
            "Hispanic": 1.04,
            "Native American": 1.06,
            "Pacific Islander": 1.03,
            "Other": 1.00
        }
        
        # Maximum risk score (prevents reaching 1.0 in most cases)
        self.max_risk_score = 0.95
        
    def calculate_frequency_risk(self, user_age, actual_visits, expected_visits, drug_name):
        """
        Calculate risk based on pharmacy visit frequency.
        
        Parameters:
        - user_age: Age of the user
        - actual_visits: Number of pharmacy visits in last 30 days
        - expected_visits: Expected number of visits in last 30 days
        - drug_name: Name of the drug being prescribed
        
        Returns:
        - Frequency risk score between 0 and 1
        """
        # Determine age factor (higher risk for younger individuals)
        if user_age < 25:
            age_factor = 1.2  # Reduced from 1.3
        elif user_age < 35:
            age_factor = 1.15  # Reduced from 1.2
        elif user_age < 50:
            age_factor = 1.05  # Reduced from 1.1
        else:
            age_factor = 1.0
            
        # If expected visits is set to 0, use a reasonable default based on drug type
        if expected_visits == 0:
            # For high-risk drugs, set default expected to 1
            high_risk_drugs = ['Heroin', 'Fentanyl', 'Crack', 'Methamphetamine']
            if drug_name in high_risk_drugs:
                expected_visits = 1
            else:
                expected_visits = 2
        
        # Calculate deviation from expected visits
        visit_deviation = actual_visits - expected_visits
            
        # Apply drug weight factor - reduced impact
        drug_weight = self.drug_weights.get(drug_name, 1.0)
        
        # Calculate base frequency risk
        if visit_deviation <= 0:
            # At or below expected - minimal risk
            frequency_risk = 0.05
        elif visit_deviation == 1:
            # Just 1 visit above expected
            frequency_risk = 0.12  # Reduced from 0.15
        elif visit_deviation == 2:
            # 2 visits above expected
            frequency_risk = 0.20  # Reduced from 0.25
        elif visit_deviation <= 4:
            # 3-4 visits above expected
            frequency_risk = 0.35  # Reduced from 0.4
        else:
            # 5+ more visits than expected - capped at 0.80
            frequency_risk = 0.45 + min(0.35, (visit_deviation - 4) * 0.06)
            
        # Apply age and drug factors with dampened effect
        frequency_risk = min(0.85, frequency_risk * age_factor * (drug_weight / 2.5))
        
        return frequency_risk
    
    def apply_demographic_factors(self, base_risk, gender, race, age):
        """
        Apply demographic risk factors to the base risk.
        
        Parameters:
        - base_risk: The initial calculated risk
        - gender: User's gender
        - race: User's race
        - age: User's age
        
        Returns:
        - Adjusted risk score between 0 and 1
        """
        # Get gender weight
        gender_weight = self.gender_weights.get(gender, 1.0)
        
        # Get race weight
        race_weight = self.race_weights.get(race, 1.0)
        
        # Calculate age weight - reduced impact
        if age < 18:
            age_weight = 0.8  # Increased from 0.7 (less extreme reduction)
        elif age < 25:
            age_weight = 1.2  # Reduced from 1.3
        elif age < 35:
            age_weight = 1.1  # Reduced from 1.2
        elif age < 50:
            age_weight = 1.05  # Reduced from 1.1
        else:
            age_weight = 1.0
        
        # Apply demographic weights
        adjusted_risk = base_risk * gender_weight * race_weight * age_weight
        
        # Ensure risk stays between 0 and maximum risk score
        return min(self.max_risk_score, adjusted_risk)
    
    def calculate_direct_risk(self, drug_name, usage_level):
        """
        Calculate direct risk based on drug type and usage level.
        Simplified version of the approach from model 1.
        
        Parameters:
        - drug_name: Name of the drug
        - usage_level: Level of usage (0-6 scale)
        
        Returns:
        - Direct risk score between 0 and 1
        """
        # Get drug weight
        drug_weight = self.drug_weights.get(drug_name, 1.0)
        
        # If the drug is not used, risk is 0
        if usage_level == 0:
            return 0.0
            
        # For daily use (levels 5-6), high base risk but less extreme
        if usage_level >= 5:
            # Start with high base risk for daily use
            base_risk = 0.65  # Reduced from 0.7
            # Add additional risk based on drug weight
            additional_risk = 0.25 * (drug_weight / 3.5)  # Reduced from 0.3
            risk = min(0.90, base_risk + additional_risk)
        else:
            # For non-daily use, gradual risk scale
            # Apply exponential scaling to usage with drug weight
            risk = (usage_level / 6.0) * drug_weight * (1.08 ** usage_level) / 3.5  # Reduced from 1.1
            # Cap at 0.60 for non-daily use
            risk = min(0.60, risk)  # Reduced from 0.65
            
        return risk
    
    def calculate_ml_based_risk(self, drug_use_data, usage_frequency=None):
        """
        Calculate risk based on the approach from model 2, with focus on repetition patterns
        and number of substances used.
        
        Parameters:
        - drug_use_data: Dictionary with drug use information
        - usage_frequency: Dictionary mapping drugs to their usage frequency (0-6 scale)
        
        Returns:
        - ML-based risk score between 0 and 1
        """
        # Initialize usage frequency if not provided
        if usage_frequency is None:
            usage_frequency = {}
            
        # Count number of substances used
        substances_used = sum(1 for drug, used in drug_use_data.items() if used)
        
        # If no substances are used, risk is zero
        if substances_used == 0:
            return 0.0
        
        # IMPROVED: Base risk on number of substances with a more gradual scale
        if substances_used == 1:
            substances_base_risk = 0.15  # Single substance - low risk
        elif substances_used <= 3:
            substances_base_risk = 0.30  # 2-3 substances - moderate risk
        elif substances_used <= 5:
            substances_base_risk = 0.50  # 4-5 substances - high risk
        else:
            substances_base_risk = 0.70  # 6+ substances - very high risk
            
        # Identify high-risk substances
        high_risk_drugs = ['Heroin', 'Fentanyl', 'Crack', 'Methamphetamine', 'Cocaine']
        
        # Calculate repetition-based risk
        # Focus on how frequently drugs are used rather than just the count
        repetition_score = 0
        drug_count = 0
        high_risk_count = 0
        daily_use_count = 0
        
        for drug, used in drug_use_data.items():
            if not used:
                continue
                
            drug_count += 1
            # Get usage frequency (default to 1 if not specified)
            freq = usage_frequency.get(drug, 1)
            
            # Count high risk drugs
            if drug in high_risk_drugs:
                high_risk_count += 1
                
            # Count daily use drugs
            if freq >= 5:  # Daily use
                daily_use_count += 1
            
            # Higher frequencies increase the score but less dramatically
            if freq >= 5:  # Daily use
                repetition_score += 2.0  # Reduced from 2.5
                if drug in high_risk_drugs:
                    repetition_score += 0.5  # Additional score for high-risk daily use
            elif freq >= 3:  # Monthly use
                repetition_score += 0.8  # Reduced from 1.0
            else:  # Occasional use
                repetition_score += 0.4  # Reduced from 0.5
        
        # Calculate average repetition per drug
        if drug_count > 0:
            avg_repetition = repetition_score / drug_count
        else:
            avg_repetition = 0
            
        # Calculate repetition risk - less extreme scaling
        if avg_repetition > 2.0:
            # Heavy repetitive use
            repetition_risk = 0.60  # Reduced from 0.65
        elif avg_repetition > 1.5:
            # Moderate to heavy repetitive use
            repetition_risk = 0.45  # Reduced from 0.5
        elif avg_repetition > 1.0:
            # Moderate repetitive use
            repetition_risk = 0.30  # Reduced from 0.35
        else:
            # Light repetitive use
            repetition_risk = 0.15  # Reduced from 0.2
            
        # Calculate high risk adjustment
        if high_risk_count >= 2:
            high_risk_adjustment = 0.12
        elif high_risk_count == 1:
            high_risk_adjustment = 0.08
        else:
            high_risk_adjustment = 0
            
        # Daily use adjustment
        daily_use_adjustment = min(0.10, daily_use_count * 0.03)
            
        # Calculate final ML-based risk as weighted combination
        # This balances substance count, repetition patterns, and high-risk drug use
        ml_risk = (
            0.4 * substances_base_risk + 
            0.4 * repetition_risk + 
            high_risk_adjustment +
            daily_use_adjustment
        )
        
        # Cap the final ML risk
        return min(0.85, ml_risk)
    
    def categorize_risk(self, risk_index):
        """
        Categorize risk index into risk levels.
        
        Parameters:
        - risk_index: Risk index between 0 and 1
        
        Returns:
        - Risk level as a string
        """
        if risk_index < 0.2:
            return "Very Low"
        elif risk_index < 0.4:
            return "Low"
        elif risk_index < 0.6:
            return "Moderate"
        elif risk_index < 0.8:
            return "High"
        else:
            return "Very High"
    
    def predict_risk(self, person_data):
        """
        Predict substance abuse risk for a person based on comprehensive factors.
        
        Parameters:
        - person_data: Dictionary with person information
        
        Returns:
        - Dictionary with risk assessment
        """
        # Extract basic information
        name = person_data.get('name', 'Anonymous')
        gender = person_data.get('gender', 'Other')
        race = person_data.get('race', 'Other')
        age = person_data.get('age', 30)
        drug_name = person_data.get('drug_name', 'Other')
        actual_visits = person_data.get('actual_visits', 0)
        expected_visits = person_data.get('expected_visits', 0)
        
        # Get drug usage information
        usage_level_str = person_data.get('usage_level', 'Never used')
        usage_level = self.usage_scale.get(usage_level_str, 0)
        
        # 1. Calculate direct risk based on drug and usage level
        direct_risk = self.calculate_direct_risk(drug_name, usage_level)
        
        # 2. Calculate frequency risk based on pharmacy visits
        frequency_risk = self.calculate_frequency_risk(
            age, actual_visits, expected_visits, drug_name
        )
        
        # 3. Prepare for ML-based risk with frequency information
        # Create drug use data dictionary
        drug_use_data = {
            'Alcohol': person_data.get('alcohol_use', False),
            'Cannabis': person_data.get('marijuana_use', False),
            'Cocaine': person_data.get('cocaine_use', False),
            'Methamphetamine': person_data.get('meth_use', False),
            'Opioid Pain Relievers': person_data.get('opioid_use', False),
            'Stimulants': person_data.get('stimulant_use', False),
            'Fentanyl': person_data.get('fentanyl_use', False)
        }
        
        # If the main drug is used, add it to the drug use data
        if usage_level > 0:
            drug_use_data[drug_name] = True
        
        # Create usage frequency dictionary for repetition analysis
        usage_frequency = {}
        
        # Assign frequency values based on user input
        # This maps boolean responses to conservative frequency estimates
        if person_data.get('alcohol_use', False):
            usage_frequency['Alcohol'] = person_data.get('alcohol_freq', 3)  # Default to monthly use
        
        if person_data.get('marijuana_use', False):
            usage_frequency['Cannabis'] = person_data.get('marijuana_freq', 2)  # Default to yearly use
            
        if person_data.get('cocaine_use', False):
            usage_frequency['Cocaine'] = person_data.get('cocaine_freq', 2)  # Default to yearly use
            
        if person_data.get('meth_use', False):
            usage_frequency['Methamphetamine'] = person_data.get('meth_freq', 2)  # Default to yearly use
            
        if person_data.get('opioid_use', False):
            usage_frequency['Opioid Pain Relievers'] = person_data.get('opioid_freq', 3)  # Default to monthly use
            
        if person_data.get('stimulant_use', False):
            usage_frequency['Stimulants'] = person_data.get('stimulant_freq', 3)  # Default to monthly use
            
        if person_data.get('fentanyl_use', False):
            usage_frequency['Fentanyl'] = person_data.get('fentanyl_freq', 3)  # Default to monthly use
        
        # Add the primary drug with its actual usage level
        if usage_level > 0:
            usage_frequency[drug_name] = usage_level
        
        # Calculate ML-based risk with frequency information
        ml_risk = self.calculate_ml_based_risk(drug_use_data, usage_frequency)
        
        # 4. Calculate combined base risk (weighted average)
        # Higher weight for visit frequency when actual visits > expected
        if actual_visits > expected_visits:
            base_risk = (
                0.4 * direct_risk +     # Direct risk from model 1
                0.3 * ml_risk +         # ML-based risk from model 2 (increased weight)
                0.3 * frequency_risk    # Slightly reduced weight for pharmacy visit frequency
            )
        else:
            base_risk = (
                0.45 * direct_risk +    # Direct risk from model 1
                0.4 * ml_risk +         # ML-based risk from model 2 (increased weight)
                0.15 * frequency_risk   # Lower weight when visits are as expected
            )
        
        # 5. Apply demographic factors
        final_risk = self.apply_demographic_factors(base_risk, gender, race, age)
        
        # 6. Get risk category
        risk_level = self.categorize_risk(final_risk)
        
        # 7. Create detailed response
        result = {
            'name': name,
            'risk_index': round(final_risk, 4),
            'risk_level': risk_level,
            'components': {
                'direct_risk': round(direct_risk, 4),
                'frequency_risk': round(frequency_risk, 4),
                'ml_risk': round(ml_risk, 4),
                'base_risk': round(base_risk, 4)
            },
            'assessment_date': datetime.now().strftime("%Y-%m-%d")
        }
        
        return result


# Interactive interface for the model
def get_user_input():
    """
    Get user input for substance abuse risk prediction.
    
    Returns:
    - Dictionary with user information
    """
    print("\nSubstance Abuse Risk Assessment\n" + "-" * 30)
    
    person_data = {}
    
    # Basic information
    person_data['name'] = input("Enter person name: ")
    
    # Gender
    print("\nGender options: Male, Female, Other")
    person_data['gender'] = input("Enter gender: ")
    
    # Race
    print("\nRace options: White, Black, Asian, Hispanic, Native American, Pacific Islander, Other")
    person_data['race'] = input("Enter race: ")
    
    # Age
    try:
        person_data['age'] = int(input("Enter age: "))
    except ValueError:
        print("Invalid age. Using default value 30.")
        person_data['age'] = 30
    
    # Drug information
    person_data['drug_name'] = input("Enter primary drug name: ")
    
    # Usage level
    print("\nUsage scale:")
    print("0: Never used")
    print("1: Used only once")
    print("2: Used in last year")
    print("3: Used in last month")
    print("4: Used in last week")
    print("5: Used in last day")
    print("6: Used multiple times daily")
    
    try:
        usage_level = int(input("Enter usage level (0-6): "))
        usage_level = max(0, min(6, usage_level))  # Ensure value is between 0-6
        
        # Convert numeric scale to string description
        usage_descriptions = {
            0: "Never used",
            1: "Used only once",
            2: "Used in last year",
            3: "Used in last month",
            4: "Used in last week",
            5: "Used in last day",
            6: "Used multiple times daily"
        }
        person_data['usage_level'] = usage_descriptions[usage_level]
    except ValueError:
        print("Invalid usage level. Using default 'Never used'.")
        person_data['usage_level'] = "Never used"
    
    # Pharmacy visits
    try:
        person_data['actual_visits'] = int(input("Enter number of pharmacy visits in last 30 days: "))
    except ValueError:
        print("Invalid input. Using default value 0.")
        person_data['actual_visits'] = 0
        
    try:
        person_data['expected_visits'] = int(input("Enter anticipated number of pharmacy visits in last 30 days: "))
    except ValueError:
        print("Invalid input. Using default value 0.")
        person_data['expected_visits'] = 0
    
    # Additional drug use information
    print("\nAdditional substance use information (answer yes/no):")
    person_data['alcohol_use'] = input("Alcohol use in past month? ").lower() in ['yes', 'y', 'true', '1']
    person_data['marijuana_use'] = input("Marijuana use in past year? ").lower() in ['yes', 'y', 'true', '1']
    person_data['cocaine_use'] = input("Cocaine use in past year? ").lower() in ['yes', 'y', 'true', '1']
    person_data['meth_use'] = input("Methamphetamine use in past year? ").lower() in ['yes', 'y', 'true', '1']
    person_data['opioid_use'] = input("Opioid misuse in past month? ").lower() in ['yes', 'y', 'true', '1']
    person_data['stimulant_use'] = input("Stimulant misuse in past month? ").lower() in ['yes', 'y', 'true', '1']
    person_data['fentanyl_use'] = input("Fentanyl use in past year? ").lower() in ['yes', 'y', 'true', '1']
    
    return person_data


def display_results(result):
    """
    Display substance abuse risk assessment results.
    
    Parameters:
    - result: Dictionary with risk assessment results
    """
    print("\n" + "=" * 50)
    print(f"SUBSTANCE ABUSE RISK ASSESSMENT FOR {result['name'].upper()}")
    print("=" * 50)
    
    print(f"\nRisk Index: {result['risk_index']:.4f}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    
    print("\nRisk Component Breakdown:")
    print(f"- Direct drug risk: {result['components']['direct_risk']:.4f}")
    print(f"- Pharmacy visit frequency risk: {result['components']['frequency_risk']:.4f}")
    print(f"- Multiple substance use risk: {result['components']['ml_risk']:.4f}")
    print(f"- Combined base risk: {result['components']['base_risk']:.4f}")
    
    print(f"\nAssessment Date: {result['assessment_date']}")
    
    # Interpretation
    print("\nInterpretation:")
    if result['risk_level'] == "Very Low":
        print("Minimal likelihood of problematic drug use. Current patterns suggest responsible and infrequent use if any.")
    elif result['risk_level'] == "Low":
        print("Some drug use patterns present but generally low concern. Monitor for any increase in frequency or diversity of use.")
    elif result['risk_level'] == "Moderate":
        print("Notable drug use patterns that warrant attention. Regular use indicates potential for developing problematic patterns.")
    elif result['risk_level'] == "High":
        print("Significant drug use patterns suggesting potential for dependency or abuse. Usage patterns indicate concerning behavior.")
    else:  # Very High
        print("Extensive drug use patterns indicating high probability of dependency or abuse. Intervention may be necessary.")
    
    print("\nRECOMMENDATION:")
    if result['risk_level'] in ["High", "Very High"]:
        print("- Consider referral for comprehensive substance use evaluation")
        print("- Implement more frequent monitoring")
        print("- Consider supervised medication administration")
    elif result['risk_level'] == "Moderate":
        print("- Increase monitoring frequency")
        print("- Provide education on substance abuse risks")
        print("- Consider brief intervention techniques")
    else:
        print("- Continue standard monitoring")
        print("- Provide general substance abuse prevention education")
    
    print("\nNOTE: This assessment is for screening purposes only and should be")
    print("used in conjunction with clinical judgment and comprehensive evaluation.")
    print("=" * 50)


def main():
    """
    Main function to run the substance abuse risk prediction.
    """
    # Create the predictor
    predictor = SubstanceAbuseRiskPredictor()
    
    # Get user input
    print("Welcome to the Substance Abuse Risk Assessment Tool")
    print("This tool helps identify potential substance abuse risks based on")
    print("multiple factors including drug use patterns and demographic information.")
    
    # Ask for user input
    person_data = get_user_input()
    
    # Make prediction
    result = predictor.predict_risk(person_data)
    
    # Display results
    display_results(result)
    
    # Ask if user wants to try another assessment
    another = input("\nWould you like to perform another assessment? (yes/no): ")
    if another.lower() in ['yes', 'y']:
        main()


if __name__ == "__main__":
    main()
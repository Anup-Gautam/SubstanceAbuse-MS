import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('Drug_Consumption.csv')
    print(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'Drug_Consumption.csv' not found. Please ensure the file is in the correct location and accessible.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# 1. Examine data types and inconsistencies
print("\nData Info:")
print(df.info())

# 2. Descriptive statistics of numerical features
print("\nDescriptive Statistics:")
print(df.describe())

# 3. Distribution of categorical features
categorical_cols = ['Gender', 'Education', 'Country', 'Ethnicity']
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())
    plt.figure(figsize=(8, 6))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# 4. Correlation between features related to drug usage
drug_usage_cols = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                   'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                   'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

# Map drug usage levels from 'CL0' to 'CL6' to numeric 0–6
usage_mapping = {
    'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3,
    'CL4': 4, 'CL5': 5, 'CL6': 6
}

for col in drug_usage_cols:
    if col in df.columns:
        df[col] = df[col].map(usage_mapping)

# Compute correlation matrix
numeric_drugs = df[drug_usage_cols]
corr = numeric_drugs.corr()

# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Between Drug Usage Features')
plt.tight_layout()
plt.show()

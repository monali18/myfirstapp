
pip install -U imbalanced-learn
import pandas as pd
import RandomUnderSampler

# Load the dataset into a DataFrame
df = pd.read_csv('C:/Users/SIR/Downloads/crunchbase-weiss.csv')
X = df.drop(columns=['is_acquired'])
y = df['is_acquired']
# Initialize the RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Fit and transform the dataset to perform random undersampling
X_resampled, y_resampled = rus.fit_resample(X, y)
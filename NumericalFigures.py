import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

df = pd.read_csv("lending_club_full_preprocessed.csv")
def missing_values_ratios(df):
 # Calculate the ratio of missing values in each column
 return df.isna().sum() / len(df)
st.subheader("Missing values ratios")
# Some features are not informative and should be removed? Which ones?
# List of features with a missing value ratio above 0.6
missingValues = st.write(missing_values_ratios(df))
print('Missing values::',missingValues)
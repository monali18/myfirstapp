import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder    
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.inspection import PartialDependenceDisplay

# For visualizing pipelines
from sklearn.utils import estimator_html_repr
import streamlit.components.v1 as components

@st.cache_data()
def load_data():
    return pd.read_csv("lending_club_full.csv")

st.title("Lending Club Prediction")

# DATA UNDERSTANDING AND PREPARATION
# Checklist:
# Select (or create) a target
# Create derived features from existing features
# Identify categorical features
# Identify numerical features and their correlations
# Check for look-ahead bias (data leakage)
# Check for missing values
# Check for duplicate rows
# Check for class imbalance

# Load data
df = load_data()

if st.sidebar.checkbox("Show dataset (initial)"):
    st.header('Dataset')
    st.write(df)

# Select (or create) a target

# Loan status is a binary feature
# Map loan_status to 0 (Charged Off) or 1 (Fully Paid)
df['loan_status'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

# Create derived features from existing features
# NA

# Identify categorical features
# We do this first, because some of them may convert into numerical features
# Candidates: home_ownership, grade, and purpose

# Home ownership is a categorical feature that can be turned into a binary feature
# Map home_ownership to 1 if MORTGAGE or OWN, and 0 otherwise
df['home_ownership'] = df['home_ownership'].map({'MORTGAGE': 1, 'OWN': 1, 'RENT': 0, 'OTHER': 0, 'NONE': 0})

# Grade is an ordinal feature, ie the values can be compared
# A is the best grade, and G is the worst grade
# Map grade to 7 if A, 6 if B, 5 if C, 4 if D, 3 if E, 2 if F, and 1 if G
df['grade'] = df['grade'].map({'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1})

# This leaves the following categorical features
categorical_features = ["purpose"]
categorical_features = []

# Identify numerical features and their correlations
numerical_features = ["loan_amount", "grade", "home_ownership", "income", "dti", "fico"]
numerical_features = ["home_ownership", "income", "dti", "fico"]
numerical_features_and_target = numerical_features + ["loan_status"]

if st.sidebar.checkbox("Show correlation matrix"):
    st.subheader("Correlation matrix")
    # Show the correlations of the numerical features with one another and with the target
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(df[numerical_features_and_target].corr(), annot=True, fmt=".2f", ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Correlations of numerical features and target")
    st.pyplot(fig)

# Check for look-ahead bias (data leakage)
# NA

# Check for missing values
def missing_values_ratios(df):
    # Calculate the ratio of missing values in each column
    return df.isna().sum() / len(df)

if st.sidebar.checkbox("Show missing values ratios"):
    st.subheader("Missing values ratios")
    # Some features are not informative and should be removed? Which ones?
    # List of features with a missing value ratio above 0.6
    st.write(missing_values_ratios(df))

# drop rows with missing dti values
df = df.dropna(subset=['dti'])

# Check for duplicate rows
df = df.drop_duplicates()

if st.sidebar.checkbox("Show dataset (after pre-processing)"):
    st.header('Dataset (after pre-processing)')
    st.write(df)

# Check for class imbalance
if st.sidebar.checkbox("Show class imbalance"):
    st.subheader("Class imbalance")
    # Show the class imbalance
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    df["loan_status"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class imbalance")
    ax.set_xlabel("Loan status")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Ratio of good loans to bad loans
    counts = df["loan_status"].value_counts()
    st.markdown(f"Ratio of good loans to bad loans: {counts[1]/counts[0]:.2f} : 1")

# MODELING
# Checklist:
# Create a pipeline for pre-processing features
# Create a pipeline for training the model

# Create a pipeline for pre-processing features
def pre_processor(numerical_features, categorical_features):
    # Pipeline for pre-processing numerical features
    # Scale all values to zero mean and unit variance (many algorithms assume this)
    numerical_transformer = Pipeline(        
        steps = [('scaler', StandardScaler())])
        # If we had missing values, we could use SimpleImputer here
        # steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        #     ('scaler', StandardScaler())])

    # Pipeline for pre-processing categorical features
    # One-hot encode the values
    categorical_transformer = Pipeline(
        steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))])
        # If we had missing values, we could use SimpleImputer here
        # steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        #     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine the two pipelines into one
    preprocessor = ColumnTransformer(
        transformers = [('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    return preprocessor

# Create a pipeline for training the model
preprocessor = pre_processor(numerical_features, categorical_features)
type_of_classifier = st.sidebar.radio("Select type of classifier", 
    ("Decision Tree", "Logistic Regression", "Random Forest"))
if type_of_classifier == "Decision Tree":
    classifier = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
elif type_of_classifier == "Logistic Regression":
    classifier = LogisticRegression(max_iter=2000, penalty='l2', class_weight='balanced')
elif type_of_classifier == "Random Forest":
    classifier = RandomForestClassifier(class_weight='balanced')
model = Pipeline(steps=[('preprocessor', preprocessor),
    ('classifier', classifier)])

# Show the pipeline
if st.sidebar.checkbox("Show pipeline"):
    st.subheader("Pipeline")
    components.html(estimator_html_repr(model), height=500)

# EVALUATION
# Checklist:
# Choose train and test datasets (stratified? time-split?)
# Identify the most suitable performance metrics for evaluating the model
# Choose a way to deal with imbalanced data
# Use k-fold cross-validation to evaluate the model (stratified? time-split?)
# Fit and evaluate the model against the test dataset
# Rank the features by their importance

# Get features X and target y
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Choose train and test datasets (stratified? time-split?)
# When the samples are independent, create a stratified split
# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y, random_state=42)

# Identify the most suitable performance metrics for evaluating the model
# If the classes are balanced, accuracy is a good metric
# Otherwise, most of the time, we would use the F1 score which balances precision and recall
# However, there may be better metrics for a specific problem

# Choose a way to deal with imbalanced data
# General strategies:
# We can use class weights to balance the classes
# We can also oversample the minority class or undersample the majority class
# Here, we will use class weights (class_weight='balanced')

if st.sidebar.checkbox("Use undersampling"):
    # The dataset is imbalanced: more loans are good than bad
    # Undersampling the majority class will result in a balanced dataset
    # Undersample the training data to balance the classes
    counts = y_train.value_counts()  # 0 = bad, 1 = good
    st.write(f"Training data contains {counts[0]} bad loans and {counts[1]} good loans ({counts[0] + counts[1]} loans in total)")
    minority_class = X_train[y_train==0]
    majority_class_undersampled = X_train[y_train==1].sample(counts[0], random_state=42)
    st.write(f"Undersampled training data: {majority_class_undersampled.shape[0]} samples")
    X_train = pd.concat([minority_class, majority_class_undersampled])
    y_train = y_train[X_train.index]

# Use k-fold cross-validation to evaluate the model
if st.sidebar.checkbox("Show model performance (cross validation)"):
    st.subheader('Model performance (cross validation)')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, n_jobs=-1,
        scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'])
    # Performance scores for each fold
    st.write("Scores for each fold (only positive class):")
    df_scores = pd.DataFrame(scores).transpose()
    df_scores['mean'] = df_scores.mean(axis=1)
    st.dataframe(df_scores)

# Fit and evaluate the model against the test dataset
# This model can be used to make predictions on unseen data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def show_equation(classifier):
    # Show the intercept and coefficients of the logistic regression model
    intercept = classifier.intercept_[0]
    coef = classifier.coef_[0]
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    # Write the equation of the logistic regression model in markdown
    equation = f"y = {intercept:.2f}"
    for i in range(len(coef)):
        equation += f" {'+' if coef[i] > 0 else '-'} {abs(coef[i]):.2f} ({feature_names[i]})"
    equation = equation.replace("cat__", "")
    equation = equation.replace("num__", "")
    equation = equation.replace("_", "\_")
    return equation

if type_of_classifier == 'Logistic Regression':
    st.subheader('Logistic regression coefficients')
    st.markdown(f"$${show_equation(classifier)}$$")

if st.sidebar.checkbox('Show model performance', value=True):
    st.subheader('Model performance')
    st.write(f"Performance of the {type_of_classifier} classifier on the test data.")

    # Scores from applying the model to the test dataset
    score = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(score).transpose()

    # Drop unnecessary rows
    df_report = df_report.drop(['accuracy', 'macro avg'], axis=0)

    # Visualize using a heatmap
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(df_report.iloc[:, :-1], annot=True, cbar=False, cmap=plt.cm.Blues)
    plt.title('Classification Report Heatmap')
    st.pyplot(fig)

if st.sidebar.checkbox('Show confusion matrix'):
    st.subheader('Confusion matrix')
    st.write(f"Confusion matrix for the {type_of_classifier} classifier on the test data.")
    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')
    ax.yaxis.set_ticklabels(['No', 'Yes'])
    st.pyplot(fig)

# Rank the features by their importance

# Get feature names
# Optional: setting verbose_feature_names_out to False will remove the name of the transformer
model.named_steps['preprocessor'].verbose_feature_names_out=False
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

if st.sidebar.checkbox('Show feature importance'):
    st.subheader('Feature importance')
    if type_of_classifier == 'Logistic Regression':
        st.write("Feature importance not supported by logistic regression")
    else:
        # Get feature importance from the model
        feature_importance = model.named_steps['classifier'].feature_importances_
        # Create a dataframe with the feature names and their importance
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        # Only include the top 10 features
        feature_importance = feature_importance.head(10)
        # Plot the feature importance as horizontal bar chart
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax, color='blue')
        ax.set_xlabel('Feature importance')
        ax.set_ylabel('Feature')
        ax.set_title("Feature importance")
        st.pyplot(fig)

if st.sidebar.checkbox('Show partial dependence plots'):
    st.subheader("Partial dependence plots")
    if type_of_classifier == 'Logistic Regression' or type_of_classifier == 'Neural Network':
        st.write(f"Partial dependence plots not supported for {type_of_classifier}")
    else:
        # Get partial dependence plots
        features = ['income', 'dti', 'fico']
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        # Plot partial dependence plots using PartialDependenceDisplay.from_estimator
        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_partial_dependence_visualization_api.html
        PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
        st.pyplot(fig)


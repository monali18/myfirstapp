############ Group 10. Assignment 2 ###############

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn import metrics

from sklearn.utils import estimator_html_repr
import streamlit.components.v1 as components

########## 2. Understanding and Preparation ##########

# 2.c.  Load and pre-process data
df = pd.read_csv("crunchbase.csv")
# Variables for filtering csv
greaterThanSevenYrs = 365*7
lessThanThreeYrs = 365*3
# Condition for filtering data 
filtered_data = df[(df['age'] < greaterThanSevenYrs) & (df['age'] > lessThanThreeYrs)]
df.to_csv("AgeBetween3to7years.csv")
# Count of companies
total_records = len(filtered_data)
print('Total no of companies whose age is greater than 3 years and less than 7 years is',total_records)


# 2.d.  Create a function to check if either ipo or is_aquired is True
def check_condition(row):
    return row['ipo'] or row['is_acquired']
# Use the map function to apply the function to each row
df['Success'] = df.apply(check_condition, axis=1)
df['Success'] = df['Success'].map(
 {True: 1, False: 0})
# printing the target variable
print(df['Success'])
df.to_csv("TargetVariable.csv")


# 2.e. creating derived feature of total number of degrees
# df = df.fillna(0)
df['number_degrees'] = df['mba_degree'] + df['ms_degree'] + df['phd_degree'] + df['other_degree']
print('The sum of all degrees for every company is ',df['number_degrees'])
df.to_csv("number_degree.csv")


#2.f. Numerical Features correlation in a heatmap
fig, ax = plt.subplots(figsize=(6.4, 4.8))
numerical_features_and_target = ['average_funded', 'total_rounds', 'average_participants', 'products_number','acquired_companies', 'offices', 'age','Success']
sns.heatmap(df[numerical_features_and_target].corr(), annot=True, 
 fmt=".2f", ax=ax, cmap=plt.cm.Blues)
ax.set_title("Correlations of numerical features and target")
st.pyplot(fig)



# 2.i. Missing values ratios
def missing_values_ratios(df):
 # Calculate the ratio of missing values in each column
 return df.isna().sum() / len(df)
st.subheader("Missing values ratios")
# List of features with a missing value ratio above 0.6
missingValues = st.write(missing_values_ratios(df))



# 2.j. Default values for missing values
df['products_number'] = df['products_number'].fillna(0)
df['acquired_companies'] = df['acquired_companies'].fillna(0)
df['mba_degree'] = df['mba_degree'].fillna(0)
df['phd_degree'] = df['phd_degree'].fillna(0)
df['ms_degree'] = df['ms_degree'].fillna(0)
df['number_degrees'] = df['number_degrees'].fillna(0)
missingValues = st.write(missing_values_ratios(df))

############# 3. Modelling ###################

# 3.a. Get features X and target y (y will not be part of input)
X = df.drop('Success', axis=1)
y = df['Success']
# Choose train and test datasets
# Split the data into 80% training and 20% testing, we use the stratified sampling in shuffling
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42)



 # 3.b Pipeline for pre-processing numerical and categorical features
categorical_features = ['category_code', 'country_code', 'state_code']
numerical_features = ['average_funded', 'total_rounds', 'average_participants', 
                      'products_number','acquired_companies', 'offices', 'age']
def pre_processor(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers = 
    [('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])
    return preprocessor


# 3c. Creating and Evaluating the model
preprocessor = pre_processor(numerical_features, categorical_features)
type_of_classifier = st.sidebar.radio("Select type of classifier", 
 ("Logistic Regression", "Random Forest"))
if type_of_classifier == "Logistic Regression":
 classifier = LogisticRegression(max_iter=2000, penalty='l2', class_weight='balanced')
elif type_of_classifier == "Random Forest":
 classifier = RandomForestClassifier(class_weight='balanced')
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
# for visualizing pipelines.
if st.sidebar.checkbox("Show pipeline"):
    st.subheader("Pipeline")
    components.html(estimator_html_repr(model), height=500)

############### 4. Evaluation #######################

#4.c
# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X, y, cv=cv, n_jobs=-1,
 scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'])
# Performance scores for each fold
st.write("Scores for each fold (only positive class):")
df_scores = pd.DataFrame(scores).transpose()
df_scores['mean'] = df_scores.mean(axis=1)
st.dataframe(df_scores)

#4.d
# Scores from applying the model to the test dataset and showing the performance metrics in a heatmap
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = metrics.classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(score).transpose()
# Drop unnecessary rows
df_report = df_report.drop(['accuracy', 'macro avg'], axis=0)
# Visualize using a heatmap
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sns.heatmap(df_report.iloc[:, :-1], annot=True, cbar=False, 
 cmap=plt.cm.Blues)
plt.title('Classification Report Heatmap')
st.pyplot(fig)

# The dataset is imbalanced: more loans are good than bad
# Undersampling the majority class will result in a balanced dataset
# Undersample the training data to balance the classes
counts = y_train.value_counts() # 0 = bad, 1 = good
st.write(f"Training data contains {counts[0]} failed startups and \
 {counts[1]} successful startups ({counts[0] + counts[1]} startups in total)")
minority_class = X_train[y_train==0]
majority_class_undersampled = X_train[y_train==1].sample(
 counts[0], random_state=42, replace=True)
st.write(f"Undersampled training data: \
 {majority_class_undersampled.shape[0]} samples")
X_train = pd.concat([minority_class, majority_class_undersampled])
y_train = y_train[X_train.index]
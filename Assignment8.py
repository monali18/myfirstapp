import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder    
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.inspection import PartialDependenceDisplay

# For visualizing pipelines
from sklearn.utils import estimator_html_repr
import streamlit.components.v1 as components

st.subheader('Assignment 2(TIMG 5301 - Group 8)')
st.markdown(f'<h1 style="color:#00008B;font-size:20px;">{"Raksha Chaudhary,Vindhya Kabeerdass,Sahil Rajput, Ashwani Kumar Singh"}</h1>',unsafe_allow_html=True)
@st.cache(allow_output_mutation=True)
def load_data():
    df=pd.read_csv('crunchbase.csv')
    return df
st.title('Crunchbase Data')
df=load_data()
num_comp=df.count()[0]

#************************ 2. Data understanding and preparation **********************************

#Counting the number of companies

if st.sidebar.button("Preprocessed Data"):
    st.subheader("Preprocessed Data")
    st.write(df)
    st.write('Number of companiesp initially provided:' , num_comp)

#2.a Removing all companies from the dataset that are older than 7 years or younger than 3 years of age

df=df[(df.age <2556) & (df.age > 1095)] 
num_companies=df.count()[0]

#2.b Creating a new target variable (success) with two values: 1 for success and 0 for failure. 
#Using the definition of startup success provided to determine the value of the target variable.

def success(row):
    if row['ipo'] ==True or row['is_acquired']==True and row['is_closed']==False:
        return 1
    else:
        return 0
df['success']= df.apply(success, axis=1)

df = df.drop('ipo', axis=1)
df = df.drop('is_acquired', axis=1)
df = df.drop('is_closed', axis=1)

#Dealing with missing values

df['mba_degree'] = df['mba_degree'].fillna(0)
df['phd_degree'] = df['phd_degree'].fillna(0)
df['ms_degree'] = df['ms_degree'].fillna(0)
df['other_degree'] = df['other_degree'].fillna(0)

#2.c Combining the features related to the education levels of the founders (mba_degree, phd_degree, ms_degree, other_degree)
#into a new feature for the total number of degrees obtained by the founders (number_degrees).

df['number_degrees']=""
column_names = ['mba_degree', 'phd_degree', 'ms_degree', 'other_degree']
df['number_degrees']= df[column_names].sum(axis=1)

#dropping the education level columns of founders after combining these features into number_degrees attribute.

df = df.drop('mba_degree', axis=1)
df= df.drop('phd_degree', axis=1)
df= df.drop('ms_degree', axis=1)
df= df.drop('other_degree', axis=1)


#2.d Identifying the numerical features in the dataset and showing their correlations with one another and the target in a heatmap.

numerical_features = ['average_funded', 'total_rounds', 'average_participants', 'products_number','acquired_companies', 'offices', 'age']
numerical_features_and_target = numerical_features + ['success']

#Creating the heatmap using corelation matrix. 

if st.sidebar.button("Show correlation matrix"):
    st.subheader("Correlation matrix")                                                                                                    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  
    sns.heatmap(df[numerical_features_and_target].corr(), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlations of numerical features and target")
    st.pyplot(fig)
    
#2.e Identifying the categorical features in the dataset

categorical_features = ['category_code', 'country_code', 'state_code']

# 2.g Computing the missing values ratio for all features, ie the count of missing values (NA) in a column over the number of values

def missing_values_ratios(df):
    return df.isna().sum() / len(df) 
if st.sidebar.button("Show missing values ratios"):
    st.subheader("Missing values ratios")
    st.write(missing_values_ratios(df)) 
    
# Count the number of duplicate rows

num_duplicate_rows = df.duplicated().sum()

if st.sidebar.button("Duplicate rows"):
    st.subheader('Number of duplicate rows')
    st.write("%d rows are duplicates" % (num_duplicate_rows))
    
if st.sidebar.button("Processed Data"):
    st.subheader("Processed Data")
    st.write(df)
    st.write('Number of companies that remain:', num_companies) 

    
#************************************************ 3. Modelling *******************************************

# 3.c Creating a pipeline for pre-processing numerical and categorical features

def pre_processor(numerical_features, categorical_features):

    numerical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
    ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers = 
    [('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])
    return preprocessor



# 3.b,c,d,and e pipeline for training the models, Decision tree and RandomForest Classifiers 

preprocessor = pre_processor(numerical_features, categorical_features)
type_of_classifier = st.sidebar.radio("Select type of classifier", ("Decision Tree", "RandomForestClassifier"))

if type_of_classifier == "Decision Tree":
    classifier = DecisionTreeClassifier(random_state = 1, max_depth = 14, min_samples_split = 2, min_samples_leaf = 1,class_weight='balanced',)
elif type_of_classifier == "RandomForestClassifier":
    classifier = RandomForestClassifier(random_state = 1, max_depth = 15,  n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1,class_weight='balanced', n_jobs=-1)
model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifier)])


# Get features X and target y (y will not be part of input)

X = df.drop('success', axis=1)
y = df['success']


# Choosing train and test datasets
# Split the data into 70% training and 30% testing, we use the stratified sampling in shuffling because our sample is not orderd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)



#Checking for Imbalance

counts = df['success'].value_counts()
fig, ax = plt.subplots(figsize=(6.4, 2.4))
sns.barplot(x=counts.index, y=counts.values, ax=ax)
ax.set_xticklabels(['No', 'Yes'])
ax.set_xlabel('Subscribed to term deposit')
ax.set_ylabel('Number of clients')
ax.set_title('Class imbalance')

subscribers = counts[1]
non_subscribers = counts[0]

if st.sidebar.button('Check imbalance'):
    st.subheader('Class imbalance')
    st.pyplot(fig)
    st.write('Degree of imbalance: %.1f to 1 non-subscribers to subscribers' % (non_subscribers/subscribers))
    st.write(subscribers)
    st.write(non_subscribers)
    
    
#************************************************** 4.Evaluation *******************************************

#4.b,c Removing imbalance using k-fold cross-validation and evaluating the model

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_train, y_train, scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'], cv=cv, n_jobs=-1)
if st.sidebar.button("Cross validation Performance"):
    st.subheader('Cross Validation Performance')
# Performance scores for each fold
    st.write("Scores for each fold (only positive class):")
    df_scores = pd.DataFrame(scores).transpose()
    df_scores['mean'] = df_scores.mean(axis=1)
    st.dataframe(df_scores)
    
    
# 4.d Evaluate the model against the test dataset
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

if st.sidebar.button('Model Performance'):
    st.subheader('Model performance')
    st.write(f"The table shows the performance of the {type_of_classifier} classifier on the test data.")
    # Scores from applying the model to the test dataset
    score = metrics.classification_report(y_test, y_pred, output_dict=True)
    # Add AUC score to the report
    score['auc'] = metrics.roc_auc_score(y_test, y_pred)
    score = pd.DataFrame(score).transpose()
    st.write("Scores from applying the model to the test dataset:")
    st.write(score)
    
if st.sidebar.button('Confusion matrix'):
    st.subheader('Confusion matrix')
    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Greens)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes'])
    ax.yaxis.set_ticklabels(['No', 'Yes'])
    st.pyplot(fig)
    
    

# 4.e Ranking the features by their importance


# Optional: setting verbose_feature_names_out to False will remove the name of the transformer

#There could be an AttributeError sometimes while fetching the features. Please run following command to install additional packages from scikit to fix the issue: 
#pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn -U
model.named_steps['preprocessor'].verbose_feature_names_out=False
feature_name = model.named_steps['preprocessor'].get_feature_names_out()

if st.sidebar.button('Feature importance'):
    # Get feature importance from the model
    feature_importance = model.named_steps['classifier'].feature_importances_
    # dataframe showing the feature names and their importance
    feature_importance = pd.DataFrame({'feature': feature_name, 'importance': model['classifier'].feature_importances_})
    #sort the list in descending order
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    # Only show the first 10 features
    feature_importance = feature_importance.head(15)
    # Plot the feature importance as horizontal bar chart
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax, color='red')
    ax.set_xlabel('Feature importance')
    ax.set_ylabel('Feature')
    ax.set_title("Feature importance")
    st.pyplot(fig)
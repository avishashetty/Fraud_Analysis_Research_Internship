import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, classification_report
from sklearn import metrics
import streamlit as st

# Reading the CSV file into a DataFrame
df = pd.read_csv('fraudTrain.csv')

# Removing missing values from the DataFrame
df_cleaned = df.dropna()

# Displaying the count of 'is_fraud' values in the cleaned DataFrame
print("Count of 'is_fraud' column")
print(df_cleaned['is_fraud'].value_counts())

# Selecting numeric and non-numeric columns from the cleaned DataFrame
numeric_df = df_cleaned.select_dtypes(include=['int64', 'float64'])
non_numeric_df = df_cleaned.select_dtypes(exclude=['int64', 'float64'])

# Scaling the numeric features using Min-Max scaling
scaler = MinMaxScaler()
scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

# Label encoding on non numeric features
label_encoder_dict = defaultdict(LabelEncoder)
encoded_data = non_numeric_df.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))

# Combining the encoded and scaled datasets
combined_df = pd.concat([encoded_data, scaled_numeric_df], axis=1)

# Resampling the data using SMOTE to address class imbalance
X = combined_df.drop('is_fraud', axis=1)
y = combined_df['is_fraud']
sm = SMOTE(random_state=2)
X_resampled, y_resampled = sm.fit_resample(X, y.ravel())

# Creating a balanced DataFrame after resampling
balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['is_fraud'])], axis=1)

# Dropping specified columns from the balanced DataFrame
print("Balanced DataFrame Columns before dropping columns")
print(balanced.columns.tolist())
balanced_df = balanced.drop(['merch_lat', 'merch_long', 'lat', 'long', 'Unnamed: 0', 'trans_date_trans_time'], axis=1)
print("Balanced DataFrame Columns after dropping columns")
print(balanced_df.columns.tolist())

# Displaying the count of 'is_fraud' values in the balanced dataset
print("Is fraud count on the balanced dataset")
print(balanced_df['is_fraud'].value_counts())
print(balanced_df.iloc[2449])

# Split the balanced data into features (X) and target (y)
X = balanced_df.drop('is_fraud', axis=1)
y = balanced_df['is_fraud']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
from xgboost import XGBClassifier
params = {'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 200, 'subsample': 0.9, 'objective': 'binary:logistic'}
model = XGBClassifier(**params)
model.fit(X_train, y_train)



# Streamlit app for real-time fraud detection
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # Get input feature values
    features = np.array(input_df_lst,dtype=np.float64).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features)
    # Display result
    if prediction[0] == 0:
        print("Legitimate transaction")
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
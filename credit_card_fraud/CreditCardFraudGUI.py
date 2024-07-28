import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score, ConfusionMatrixDisplay,classification_report
from scipy.stats import randint
from sklearn import metrics
from tkinter import messagebox
from tkinter import ttk


# Reading the CSV file into a DataFrame
df = pd.read_csv('fraudTrain.csv')

#Preprocessing steps:

# Removing missing values from the DataFrame
df_cleaned = df.dropna()

print("Count of 'is_fraud' column")
print(df_cleaned['is_fraud'].value_counts())

# Selecting numeric and non-numeric columns from the cleaned DataFrame
numeric_df = df_cleaned.select_dtypes(include=['int64','float64']).drop('is_fraud', axis=1)
non_numeric_df = df_cleaned.select_dtypes(exclude=['int64','float64'])

# Scaling the numeric features using Min-Max scaling
scaler = MinMaxScaler()
scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
joblib.dump(scaler, 'scaler.pkl')

# Label encoding on non numeric features
label_encoder_dict = defaultdict(LabelEncoder)
encoded_data = non_numeric_df.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))

for column, encoder in label_encoder_dict.items():
    joblib.dump(encoder, f'label_encoder_{column}.pkl')

# Combining the encoded and scaled datasets
combined_df = pd.concat([encoded_data,scaled_numeric_df,df_cleaned['is_fraud']] , axis=1)

# Resampling the data using SMOTE to address class imbalance
X=combined_df.drop('is_fraud',axis=1)
y=combined_df['is_fraud']
sm=SMOTE(random_state=2)
X_resampled,y_resampled =sm.fit_resample(X,y.ravel())

# Creating a balanced DataFrame after resampling
balanced=pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                       pd.DataFrame(y_resampled, columns=['is_fraud'])] , axis=1)

# Dropping specified columns from the balanced DataFrame
print("Balanced DataFrame Columns before dropping columns")
print(balanced.columns.tolist())
balanced_df=balanced.drop(['merch_lat','merch_long','lat','long','Unnamed: 0','trans_date_trans_time'],axis=1)
print("Balanced DataFrame Columns after dropping columns")
print(balanced_df.columns.tolist())

# Displaying the count of 'is_fraud' values in the balanced dataset
print("Is fraud count on the balanced dataset")
print(balanced_df['is_fraud'].value_counts())


# Split the balanced data into features (X) and target (y)
X = balanced_df.drop('is_fraud', axis=1)
y = balanced_df['is_fraud']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#xgboost model is consider since it had highest accuracy

# Importing XGBoost
from xgboost import XGBClassifier

# Specify XGBoost parameters
params = {
    'learning_rate': 0.2,
    'max_depth': 2,
    'n_estimators': 200,
    'subsample': 0.9,
    'objective': 'binary:logistic'
}

# Instantiate XGBoost model
xgb_model = XGBClassifier(**params)
# Fit the model on the sampled training set
xgb_model.fit(X_train, y_train)

# Save the trained XGBoost model
joblib.dump(xgb_model, 'credit_card_fraud_xgb_model.pkl')

# Load the saved model
model = joblib.load('credit_card_fraud_xgb_model.pkl')

# Load or create the scaler and encoders as they were during training
scaler = joblib.load('scaler.pkl')  
label_encoder_dict = defaultdict(LabelEncoder) 

# Load each label encoder
for column in non_numeric_df.columns:
    label_encoder_dict[column] = joblib.load(f'label_encoder_{column}.pkl')

# Preprocess the input data
def preprocess_input(raw_data):
    input_df = pd.DataFrame([raw_data])
    numeric_df = input_df.select_dtypes(include=['int64', 'float64'])
    non_numeric_df = input_df.select_dtypes(exclude=['int64', 'float64'])
    scaled_numeric_df = pd.DataFrame(scaler.transform(numeric_df), columns=numeric_df.columns)
    encoded_data = non_numeric_df.apply(lambda x: label_encoder_dict[x.name].transform(x))
    preprocessed_data1 = pd.concat([encoded_data, scaled_numeric_df], axis=1)
    preprocessed_data=preprocessed_data1.drop(['merch_lat','merch_long','lat','long','Unnamed: 0','trans_date_trans_time'],axis=1)
    return preprocessed_data

# Predict fraud based on user input
def predict_fraud():
    raw_data = {
        'trans_date_trans_time': trans_date_trans_time_var.get(),  
        'merchant': merchant_var.get(),
        'category': category_var.get(),
        'first': first_var.get(),
        'last': last_var.get(),
        'gender': gender_var.get(),
        'street': street_var.get(),
        'city': city_var.get(),
        'state': state_var.get(),
        'job': job_var.get(),
        'dob': dob_var.get(),
        'trans_num': trans_num_var.get(),
        'Unnamed: 0': int(unnamed_var.get()),  
        'cc_num': int(cc_num_var.get()),
        'amt': float(amt_var.get()),
        'zip': int(zip_var.get()),
        'lat': float(lat_var.get()),  
        'long': float(long_var.get()),  
        'city_pop': int(city_pop_var.get()),
        'unix_time': int(unix_time_var.get()),
        'merch_lat': float(merch_lat_var.get()),  
        'merch_long': float(merch_long_var.get())  
    }

    # Preprocess the input
    preprocessed_data = preprocess_input(raw_data)
    prediction = model.predict(preprocessed_data)

    print("Preprocessed Input Data:")
    print(preprocessed_data)

    print("Model Prediction:")
    print(prediction)

    # Show the result in a message box
    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    messagebox.showinfo("Prediction Result", result)

# GUI setup

root = tk.Tk()
root.title("Fraud Detection Prediction")
root.configure(bg="#f0f0f0")  # Set a background color

# Define styles
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
style.configure("TEntry", font=("Arial", 12))
style.configure("TButton", font=("Arial", 12, "bold"), background="#007BFF", foreground="white")

# Define GUI elements (input fields, buttons)
tk.Label(root, text="Credit Card Number:").grid(row=0, column=0)
cc_num_var = tk.StringVar()
tk.Entry(root, textvariable=cc_num_var).grid(row=0, column=1)

tk.Label(root, text="Merchant:").grid(row=1, column=0)
merchant_var = tk.StringVar()
tk.Entry(root, textvariable=merchant_var).grid(row=1, column=1)

tk.Label(root, text="Category:").grid(row=2, column=0)
category_var = tk.StringVar()
tk.Entry(root, textvariable=category_var).grid(row=2, column=1)

tk.Label(root, text="Amount:").grid(row=3, column=0)
amt_var = tk.StringVar()
tk.Entry(root, textvariable=amt_var).grid(row=3, column=1)

tk.Label(root, text="First Name:").grid(row=4, column=0)
first_var = tk.StringVar()
tk.Entry(root, textvariable=first_var).grid(row=4, column=1)

tk.Label(root, text="Last Name:").grid(row=5, column=0)
last_var = tk.StringVar()
tk.Entry(root, textvariable=last_var).grid(row=5, column=1)

tk.Label(root, text="Gender (M/F):").grid(row=6, column=0)
gender_var = tk.StringVar()
tk.Entry(root, textvariable=gender_var).grid(row=6, column=1)

tk.Label(root, text="Street:").grid(row=7, column=0)
street_var = tk.StringVar()
tk.Entry(root, textvariable=street_var).grid(row=7, column=1)

tk.Label(root, text="City:").grid(row=8, column=0)
city_var = tk.StringVar()
tk.Entry(root, textvariable=city_var).grid(row=8, column=1)

tk.Label(root, text="State:").grid(row=9, column=0)
state_var = tk.StringVar()
tk.Entry(root, textvariable=state_var).grid(row=9, column=1)

tk.Label(root, text="ZIP Code:").grid(row=10, column=0)
zip_var = tk.StringVar()
tk.Entry(root, textvariable=zip_var).grid(row=10, column=1)

tk.Label(root, text="City Population:").grid(row=11, column=0)
city_pop_var = tk.StringVar()
tk.Entry(root, textvariable=city_pop_var).grid(row=11, column=1)

tk.Label(root, text="Job:").grid(row=12, column=0)
job_var = tk.StringVar()
tk.Entry(root, textvariable=job_var).grid(row=12, column=1)

tk.Label(root, text="Date of Birth (DOB):").grid(row=13, column=0)
dob_var = tk.StringVar()
tk.Entry(root, textvariable=dob_var).grid(row=13, column=1)

tk.Label(root, text="Transaction Number:").grid(row=14, column=0)
trans_num_var = tk.StringVar()
tk.Entry(root, textvariable=trans_num_var).grid(row=14, column=1)

tk.Label(root, text="Unix Time:").grid(row=15, column=0)
unix_time_var = tk.StringVar()
tk.Entry(root, textvariable=unix_time_var).grid(row=15, column=1)

tk.Label(root, text="Transaction Date and Time:").grid(row=16, column=0)
trans_date_trans_time_var = tk.StringVar()
tk.Entry(root, textvariable=trans_date_trans_time_var).grid(row=16, column=1)

tk.Label(root, text="Unnamed (ID):").grid(row=17, column=0)
unnamed_var = tk.StringVar()
tk.Entry(root, textvariable=unnamed_var).grid(row=17, column=1)

tk.Label(root, text="Latitude:").grid(row=18, column=0)
lat_var = tk.StringVar()
tk.Entry(root, textvariable=lat_var).grid(row=18, column=1)

tk.Label(root, text="Longitude:").grid(row=19, column=0)
long_var = tk.StringVar()
tk.Entry(root, textvariable=long_var).grid(row=19, column=1)

tk.Label(root, text="Merchant Latitude:").grid(row=20, column=0)
merch_lat_var = tk.StringVar()
tk.Entry(root, textvariable=merch_lat_var).grid(row=20, column=1)

tk.Label(root, text="Merchant Longitude:").grid(row=21, column=0)
merch_long_var = tk.StringVar()
tk.Entry(root, textvariable=merch_long_var).grid(row=21, column=1)

# Predict button
tk.Button(root, text="Predict Fraud", command=predict_fraud).grid(row=23, column=0, columnspan=2)

root.mainloop()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score, ConfusionMatrixDisplay,classification_report
from scipy.stats import randint
from sklearn import metrics

# Reading the CSV file into a DataFrame
df = pd.read_csv('fraudTrain.csv')

# Displaying the head, shape, and summary statistics of the original DataFrame
print("Display head of the dataframe")
print(df.head())
print("Display shape of the dataframe")
print(df.shape)
print("Display Summary of the dataframe")
print(df.describe())

# Removing missing values from the DataFrame
df_cleaned = df.dropna()

# Displaying the cleaned DataFrame
print("Dataframe after removing null or missing values if they are present")
print(df_cleaned)

# Displaying data types of each column in the cleaned DataFrame
print("Data types of each column")
print(df_cleaned.dtypes)

# Displaying the count of 'is_fraud' values in the cleaned DataFrame
print("Count of 'is_fraud' column")
print(df_cleaned['is_fraud'].value_counts())

# Selecting numeric and non-numeric columns from the cleaned DataFrame
numeric_df = df_cleaned.select_dtypes(include=['int64','float64'])
non_numeric_df = df_cleaned.select_dtypes(exclude=['int64','float64'])
print("Numeric DataFrame")
print(numeric_df)
print("Non numeric DataFrame")
print(non_numeric_df)

# Scaling the numeric features using Min-Max scaling
scaler = MinMaxScaler()
scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
print("Scaled DataFrame")
print(scaled_numeric_df)

# Label encoding on non numeric features
label_encoder_dict = defaultdict(LabelEncoder)
encoded_data = non_numeric_df.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))
print("Encoded DataFrame")
print(encoded_data)

# Combining the encoded and scaled datasets
combined_df = pd.concat([encoded_data,scaled_numeric_df] , axis=1)
print("Combined DataFrame")
print(combined_df)

# Resampling the data using SMOTE to address class imbalance
X=combined_df.drop('is_fraud',axis=1)
y=combined_df['is_fraud']
sm=SMOTE(random_state=2)
X_resampled,y_resampled =sm.fit_resample(X,y.ravel())

# Creating a balanced DataFrame after resampling
balanced=pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                       pd.DataFrame(y_resampled, columns=['is_fraud'])] , axis=1)
print("Balanced DataFrame")
print(balanced)

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

# Display test results
def display_test_results(model_name, model):
    print(model_name)
    # Prediction on the test set
    y_test_pred = model.predict(X_test)

    # Confusion matrix
    print("------------------ Confusion Matrix --------------------")
    c_matrix = confusion_matrix(y_test, y_test_pred)
    print(c_matrix)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

    # Classification report
    print("------------------ classification_report --------------------")
    print(classification_report(y_test, y_test_pred))

    # More specific classification report
    TP = c_matrix[1, 1]  # true positive
    TN = c_matrix[0, 0]  # true negatives
    FP = c_matrix[0, 1]  # false positives
    FN = c_matrix[1, 0]  # false negatives

    # Accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))

    # Sensitivity
    print("Sensitivity:", TP / float(TP + FN))

    # Specificity
    print("Specificity:", TN / float(TN + FP))

    # F1 score
    print("F1-Score:", f1_score(y_test, y_test_pred))

    # Recall
    print("Recall:", recall_score(y_test, y_test_pred))

    # Precision
    print("Precision:", precision_score(y_test, y_test_pred))

    # Predicted probability
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC
    print("------------------ ROC --------------------")
    roc_auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred_proba, drop_intermediate=False)
    
    # Plot the ROC curve
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Random Forest

# Importing random forest classifier
from sklearn.ensemble import RandomForestClassifier
# Instantiate the random forest model
random_forest_model = RandomForestClassifier(criterion="gini", 
                                            random_state=100, 
                                            max_depth=5, 
                                            min_samples_leaf=100,
                                            min_samples_split=100, 
                                            n_estimators=100)
# Fit the model on the sampled training set                                           
random_forest_model.fit(X_train, y_train)
# Display results
display_test_results("Random Forest", random_forest_model)


#decision tree classifier
# Importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# Instantiate the decision tree model
decision_tree_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=100,
                                  min_samples_split=100)
# Fit the model on the sampled training set
decision_tree_model.fit(X_train, y_train)
# Display results
display_test_results("Decision Tree", decision_tree_model)


#xgboost
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
# Display results
display_test_results("XGBoost", xgb_model)


#logistic regression
# Importing logistic regression
from sklearn.linear_model import LogisticRegression
# Instantiate the logistic regression model
logistic_model = LogisticRegression(
    C=0.01,
    penalty='l2', 
    solver='lbfgs',
    random_state=100 
)
# Fit the model on the sampled training set
logistic_model.fit(X_train, y_train)
# Display results
display_test_results("Logistic Regression", logistic_model)


#naive bayes
#Importing GaussianNB
from sklearn.naive_bayes import GaussianNB
# Instantiate the GaussianNB
nb_model = GaussianNB()
# Train the model
nb_model.fit(X_train, y_train)
# Display results
display_test_results("Naive Bayes", nb_model)
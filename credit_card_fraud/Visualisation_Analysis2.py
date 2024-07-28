import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

# Load the dataset 
df = pd.read_csv('fraudTrain.csv')


# Count the number of fraudulent and non-fraudulent transactions
fraud_counts = df['is_fraud'].value_counts()
# Create a pie chart
plt.figure(figsize=(8, 8))
colors = ['#ff9999', 'blue']  # Red for fraud, blue for non-fraud
plt.pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Credit Card Fraud:Distribution of Fraudulent vs. Non-Fraudulent Transactions')
plt.legend(['Non-Fraud', 'Fraud'])
percentages = [fraud_counts[0] / sum(fraud_counts) * 100, fraud_counts[1] / sum(fraud_counts) * 100]
labels = ['Non-Fraud ({:.1f}%)'.format(percentages[0]), 'Fraud ({:.1f}%)'.format(percentages[1])]
plt.legend(labels, loc='upper right')
plt.show()



# Filter the data for fraud = 1 and group it by 'gender'
fraudulent_data = df[df['is_fraud'] == 1]
fraud_by_gender = fraudulent_data.groupby('gender')['amt'].sum().reset_index()
# Create a bar plot
plt.figure(figsize=(12, 6))
colors = ['lightcoral', 'skyblue']
bars = plt.bar(fraud_by_gender['gender'], fraud_by_gender['amt'], color=colors)
# Adding labels to the bars with percentage values
for i, amt in enumerate(fraud_by_gender['amt']):
    plt.text(i, amt, f'{amt:.2f}', ha='center', va='bottom', fontsize=12)
plt.xlabel('Gender')
plt.ylabel('Total Fraudulent Amount in Dollars')
plt.title('Total Fraudulent Amount by Gender')
# Create a custom legend with labels and colors
legend_labels = ['Female', 'Male']
legend_colors = ['lightcoral', 'skyblue']
legend_handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
plt.legend(handles=legend_handles, loc='upper right')
# Define a function to format the y-axis labels
def y_format(x, pos):
    return f'{x:.0f}'
# Apply the formatting to the y-axis
formatter = FuncFormatter(y_format)
plt.gca().yaxis.set_major_formatter(formatter)
plt.ylim(0, max(fraud_by_gender['amt']) * 1.1)  # Set y-axis limit for proper display
plt.show()




# Filter the dataset for transactions with is_fraud = 1
fraudulent_transactions = df[df['is_fraud'] == 1]
# Group states based on the sum of amount for fraudulent transactions
state_amount_fraudulent = fraudulent_transactions.groupby('state')['amt'].sum().reset_index()
# Define the full state names
state_names = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
    'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland',
    'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada',
    'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington',
    'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
}
# Map state codes to full state names
state_amount_fraudulent['state'] = state_amount_fraudulent['state'].map(state_names)
# Sort the states by the amount in descending order and take the top 10
top_10_states = state_amount_fraudulent.sort_values(by='amt', ascending=False).head(10)
# Define different colors for each bar
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightseagreen', 'lightsalmon',
          'lightsteelblue', 'lightpink', 'lightgoldenrodyellow', 'lightcyan', 'lightblue']
# Create the bar chart with values and colors
plt.figure(figsize=(12, 6))
bars = plt.bar(top_10_states['state'], top_10_states['amt'], color=colors)
# Title and labels
plt.title('Top 10 States with Maximum Amount (Fraudulent Transactions)')
plt.ylabel('Amount  in Dollars')
plt.xlabel('State')
# Display values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom')
# Adjust x-axis label rotation for better spacing
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Filter the dataset for transactions with is_fraud = 1
fraudulent_transactions = df[df['is_fraud'] == 1]
# Group states based on the sum of amount for fraudulent transactions
state_amount_fraudulent = fraudulent_transactions.groupby('state')['amt'].sum().reset_index()
# Define the full state names
state_names = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
    'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland',
    'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada',
    'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington',
    'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
}
# Map state codes to full state names
state_amount_fraudulent['state'] = state_amount_fraudulent['state'].map(state_names)
# Find the state with the highest amount of fraudulent transactions
highest_state = state_amount_fraudulent[state_amount_fraudulent['amt'] == state_amount_fraudulent['amt'].max()]
# Find the state with the lowest amount of fraudulent transactions
lowest_state = state_amount_fraudulent[state_amount_fraudulent['amt'] == state_amount_fraudulent['amt'].min()]
# Create data for the pie chart
data = [highest_state['amt'].values[0], lowest_state['amt'].values[0]]
labels = [f"{highest_state['state'].values[0]} (Highest)", f"{lowest_state['state'].values[0]} (Lowest)"]
colors = ['skyblue', 'lightcoral']
# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('States with Highest and Lowest Amount of Fraudulent Transactions')
# Create a custom legend with the full form of states and the sum of amounts
legend_labels = [f"{highest_state['state'].values[0]} - ${highest_state['amt'].values[0]:,.2f}",
                 f"{lowest_state['state'].values[0]} - ${lowest_state['amt'].values[0]:,.2f}"]
plt.legend(legend_labels, loc='best')
plt.show()




# Filter the dataset for transactions with is_fraud = 1
fraudulent_transactions = df[df['is_fraud'] == 1]
# Group fraudulent transactions by the merchant and calculate the total amount
merchant_amount = fraudulent_transactions.groupby('merchant')['amt'].sum().reset_index()
# Sort the unique fraudulent merchants by the amount in descending order and take the top 10
top_10_fraudulent_merchants = merchant_amount.sort_values(by='amt', ascending=False).head(10)
# Define a list of different colors for the bars
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightseagreen', 'lightsalmon',
          'lightsteelblue', 'lightpink', 'lightgoldenrodyellow', 'lightcyan', 'lightblue']
# Create the bar chart with different colors
plt.figure(figsize=(12, 6))
bars = plt.bar(top_10_fraudulent_merchants['merchant'], top_10_fraudulent_merchants['amt'], color=colors)
# Title and labels
plt.title('Top 10 Fraudulent Merchants with Highest Total Amount')
plt.ylabel('Amount in Dollars')
plt.xlabel('Merchant')
# Display values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom')

# Adjust x-axis label rotation for better spacing
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
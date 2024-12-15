import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("fraudTrain.csv")
 

# Convert the 'trans_date_trans_time' column to datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
# Extract the month from the 'trans_date_trans_time' column
df['month'] = df['trans_date_trans_time'].dt.month_name()
# Filter for fraudulent transactions
fraudulent_data = df[df['is_fraud'] == 1]
# Order the months chronologically
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fraudulent_data['month'] = pd.Categorical(fraudulent_data['month'], categories=month_order, ordered=True)
# Group by month and calculate the total amount transacted
monthly_amount = fraudulent_data.groupby('month')['amt'].sum()
# Define colors for different bars
colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'lime']
# Plot the bar graph with month names and different colors
plt.bar(monthly_amount.index, monthly_amount.values, color=colors)
plt.xlabel('Month (1st Jan 2019 - 31st Dec 2020)')
plt.ylabel('Total Amount Transacted in Dollars')
plt.title('Credit Card Analysis: Total Amount Transacted Monthly')
# Display month names on the x-axis ticks in chronological order
plt.xticks(month_order, rotation=45, ha='right')
plt.show()



# Filter for fraudulent transactions
fraudulent_data = df[df['is_fraud'] == 1]
# Convert 'dob' to datetime
fraudulent_data['dob'] = pd.to_datetime(fraudulent_data['dob'])
# Calculate age using timedelta
age_bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']
# Calculate age using timedelta
age_timedelta = pd.to_datetime('today') - fraudulent_data['dob']
fraudulent_data['age_group'] = pd.cut(age_timedelta.dt.days // 365, bins=age_bins, labels=age_labels, right=False)
# Define colors for each age group
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink']
# Calculate total transaction amount for each age group
total_amount_by_age = fraudulent_data.groupby('age_group')['amt'].sum()
# Calculate the percentage for each age group
percentage_by_age = (total_amount_by_age / total_amount_by_age.sum()) * 100
# Create a bar plot with different colors and display percentages
plt.figure(figsize=(12, 6))
bars = plt.bar(total_amount_by_age.index, total_amount_by_age, color=colors)
# Add labels and title
plt.xlabel('Age Group')
plt.ylabel('Total Transaction Amount in Dollars')
plt.title('Fraudulent Transactions Across Age Groups')
# Display percentages on each bar
for bar, percentage in zip(bars, percentage_by_age):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100, f'{percentage:.2f}%', ha='center', va='bottom')
# Show the plot
plt.show()



# Apply the categorization function to create a new 'profession_category' column
df['profession_category'] = df['job'].apply(lambda job: 'Professional' if any(keyword in job.lower() for keyword in ['engineer', 'scientist', 'designer']) else 'Non-Professional')
# Filter for is_fraud=1
fraudulent_data_professional = df[df['is_fraud'] == 1]
# Grouping by profession category and summing the amounts
category_data_professional = fraudulent_data_professional.groupby('profession_category')['amt'].sum()
# Plotting the pie chart
plt.figure(figsize=(6, 6))
category_data_professional.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('Fraudulent Transaction Amount Distribution by Profession (Engineer, Scientist, Designer)')
plt.ylabel('')  # Remove y-axis label
# Adding legend
plt.legend(labels=['Professional (Engineer, Scientist, Designer)', 'Non-Professional'], loc='lower right')
plt.show()




# Define a dictionary to map states to regions
state_regions = {
    'WA': 'West', 'OR': 'West', 'CA': 'West',
    'ID': 'West', 'NV': 'West', 'AZ': 'West',
    'UT': 'West', 'MT': 'West', 'WY': 'West',
    'CO': 'West', 'NM': 'West', 'ND': 'Midwest',
    'SD': 'Midwest', 'NE': 'Midwest', 'KS': 'Midwest',
    'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest',
    'WI': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest',
    'IN': 'Midwest', 'OH': 'Midwest', 'KY': 'South',
    'TN': 'South', 'MS': 'South', 'AL': 'South',
    'GA': 'South', 'FL': 'South', 'SC': 'South',
    'NC': 'South', 'VA': 'South', 'WV': 'South',
    'MD': 'South', 'DE': 'South', 'PA': 'Northeast',
    'NJ': 'Northeast', 'NY': 'Northeast', 'CT': 'Northeast',
    'RI': 'Northeast', 'MA': 'Northeast', 'VT': 'Northeast',
    'NH': 'Northeast', 'ME': 'Northeast'
}
# Add a new column for region
df['region'] = df['state'].map(state_regions)
# Filter for fraudulent transactions
fraudulent_data_region = df[df['is_fraud'] == 1]
# Group by region and count occurrences
region_counts = fraudulent_data_region['region'].value_counts()
# Plotting the pie chart
plt.figure(figsize=(8, 8))
region_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
plt.title('Distribution of Fraudulent Transactions by Region')
plt.ylabel('')  # Remove y-axis label
plt.show()




# Filter for fraudulent transactions
fraudulent_data_category = df[df['is_fraud'] == 1]
# Group by category and calculate the total amount
category_amount = fraudulent_data_category.groupby('category')['amt'].sum()
# Sorting the data for better visualization
category_amount = category_amount.sort_values(ascending=False)
# Calculate the total sum of amounts
total_amount_category = category_amount.sum()
# Calculate the percentage for each category
percentage_by_category = (category_amount / total_amount_category) * 100
# Define colors for each category
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
# Plotting the bar chart
plt.figure(figsize=(12, 6))
category_amount.plot(kind='bar', color=colors)
plt.title('Fraudulent Transactions: Total Amount by Category')
plt.xlabel('Category')
plt.ylabel('Total Transaction Amount in Dollars')
# Display percentages on top of each bar
for i, value in enumerate(category_amount):
    plt.text(i, value + 5, f'{percentage_by_category.iloc[i]:.2f}%', ha='center')

plt.ticklabel_format(style='plain', axis='y')  
# Disable scientific notation
plt.show()
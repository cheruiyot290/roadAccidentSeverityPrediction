import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Add numpy for data types

# Load the dataset
df = pd.read_excel("data/Road Accident Data.xlsx")

# Initial number of rows
print(f"Initial number of rows: {len(df)}")

# Check for missing values
print("Initial missing values:\n", df.isnull().sum())

# Impute missing values
df['Carriageway_Hazards'] = df['Carriageway_Hazards'].fillna('Unknown')
df['Road_Surface_Conditions'] = df['Road_Surface_Conditions'].fillna(df['Road_Surface_Conditions'].mode()[0])
df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df['Weather_Conditions'] = df['Weather_Conditions'].fillna(df['Weather_Conditions'].mode()[0])
df['Time'] = df['Time'].fillna(df['Time'].mode()[0])

# Check missing values after imputation
print("Missing values after imputation:\n", df.isnull().sum())

# Number of rows after imputation
print(f"Number of rows after imputation: {len(df)}")

# Convert 'Time' to hour of day
df['Hour'] = df['Time'].apply(lambda t: t.hour if pd.notnull(t) else None)

# Verify that 'Hour' conversion is successful
print(f"Number of rows with valid 'Hour': {df['Hour'].notnull().sum()}")

# Drop rows where 'Hour' could not be converted
df = df.dropna(subset=['Hour'])
print(f"Number of rows after dropping invalid times: {len(df)}")

# Encode the target variable
severity_mapping = {'Slight': 0, 'Serious': 1, 'Fatal': 2}
df['Accident_Severity'] = df['Accident_Severity'].map(severity_mapping)

# Verify 'Accident_Severity' conversion
print(f"Unique 'Accident_Severity' values after mapping: {df['Accident_Severity'].unique()}")

# Drop rows where the target variable is NaN
df = df.dropna(subset=['Accident_Severity'])
print(f"Number of rows after dropping NaN Accident_Severity: {len(df)}")

# Define dependent and independent variables
X = df[['Day_of_Week', 'Junction_Control', 'Junction_Detail', 'Light_Conditions',
        'Local_Authority_(District)', 'Carriageway_Hazards', 'Longitude', 'Latitude',
        'Number_of_Casualties', 'Number_of_Vehicles', 'Police_Force',
        'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Hour',
        'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']]

y = df['Accident_Severity']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Check the number of samples after preprocessing
print(f"Number of samples after preprocessing: {len(X)}")

# Check for any rows where the independent variables have NaN values and drop them
X = X.dropna()
y = y[X.index]

# Check the number of samples after dropping NaN rows
print(f"Number of samples after dropping NaN rows: {len(X)}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model to a file
joblib.dump(model, 'models/road_accident_severity_model.pkl')

# Load the saved model
model = joblib.load('models/road_accident_severity_model.pkl')

# Example hypothetical data (adjust according to your data)
hypothetical_data = pd.DataFrame([{
    'Day_of_Week': 1,
    'Junction_Control': 'Automatic traffic signal',
    'Junction_Detail': 'Crossroads',
    'Light_Conditions': 'Daylight: street lights present',
    'Local_Authority_(District)': 'Westminster',
    'Carriageway_Hazards': 'Unknown',
    'Longitude': -0.143299,
    'Latitude': 51.501009,
    'Number_of_Casualties': 2,
    'Number_of_Vehicles': 2,
    'Police_Force': 1,
    'Road_Surface_Conditions': 'Dry',
    'Road_Type': 'Single carriageway',
    'Speed_limit': 30,
    'Hour': 12,
    'Urban_or_Rural_Area': 1,
    'Weather_Conditions': 'Fine no high winds',
    'Vehicle_Type': 1
}])

# Convert categorical variables to dummy/indicator variables
hypothetical_data = pd.get_dummies(hypothetical_data, drop_first=True)

# Align hypothetical_data columns with X_train columns
hypothetical_data = hypothetical_data.reindex(columns=X_train.columns, fill_value=0)

# Predict accident severity
predicted_severity = model.predict(hypothetical_data)
print(f'Predicted Accident Severity: {predicted_severity[0]}')

# Visualizations

# Distribution of Accident Severity
plt.figure(figsize=(10, 6))
sns.countplot(x='Accident_Severity', data=df, palette='viridis')
plt.title('Distribution of Accident Severity')
plt.xlabel('Accident Severity')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=['Slight', 'Serious', 'Fatal'])
plt.savefig('data/images/distribution_accident_severity.png')  # Save plot as PNG
plt.show()

# Number of Accidents by Day of the Week
plt.figure(figsize=(10, 6))
sns.countplot(x='Day_of_Week', data=df, palette='viridis')
plt.title('Number of Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.xticks(ticks=range(1, 8), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.savefig('data/images/accidents_by_day_of_week.png')  # Save plot as PNG
plt.show()

# Accident Severity vs. Speed Limit
plt.figure(figsize=(12, 8))
sns.boxplot(x='Accident_Severity', y='Speed_limit', data=df, palette='viridis')
plt.title('Accident Severity vs. Speed Limit')
plt.xlabel('Accident Severity')
plt.ylabel('Speed Limit')
plt.xticks(ticks=[0, 1, 2], labels=['Slight', 'Serious', 'Fatal'])
plt.savefig('data/images/severity_vs_speed_limit.png')  # Save plot as PNG
plt.show()

# Heatmap of Correlation Matrix
# Select numeric columns for correlation calculation
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()

# Handle warnings related to use_inf_as_na
with pd.option_context('mode.use_inf_as_na', True):
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
    plt.title('Correlation Matrix of Features')
    plt.savefig('data/images/correlation_matrix.png')  # Save plot as PNG
    plt.show()

# Pairplot of Select Features
subset_df = df[['Accident_Severity', 'Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit']]
sns.pairplot(subset_df, hue='Accident_Severity', palette='viridis')
plt.suptitle('Pairplot of Select Features', y=1.02)
plt.savefig('data/images/pairplot_features.png')  # Save plot as PNG
plt.show()

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the column data types
dtype_dict = {
    "TransactionID": "str",
    "CustomerID": "str",
    "CustomerDOB": "str",
    "CustGender": "str",
    "CustLocation": "str",
    "CustAccountBalance": "float",
    "TransactionDate": "str",
    "TransactionTime": "float",
    "TransactionAmount (INR)": "float",
    "Age": "int",
    "Recency": "float",
    "Frequency": "float",
    "Monetary": "float",
    "Cluster": "int"
}
# Load the final dataset
final_data = pd.read_csv('../data/final_customer_segmented_data.csv', dtype=dtype_dict, low_memory=False)

#scaler = MinMaxScaler()

# Calculate the time to next purchase for each customer (use Recency and Frequency for prediction)
#final_data[['Recency', 'Frequency']] = scaler.fit_transform(final_data[['Recency', 'Frequency']])
final_data['TimeToNextPurchase'] = (final_data['Recency'] * 0.5) + (final_data['Frequency'] * 30)


#final_data['TimeToNextPurchase'] = (final_data['Recency']*0.5) + (30 * final_data['Frequency'])  # Adjust the weight for Frequency

# Cap at 360 days
final_data['TimeToNextPurchase'] = final_data['TimeToNextPurchase'].apply(lambda x: min(x, 360))


# Select features (Recency, Frequency, Monetary, Cluster)
X = final_data[['Recency', 'Frequency', 'Monetary', 'Cluster']]
y = final_data['TimeToNextPurchase']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the distribution of TimeToNextPurchase (if available)

#plt.hist(final_data['TimeToNextPurchase'], bins=50, color='skyblue', alpha=0.7)
#plt.title('Distribution of Time to Next Purchase')
#plt.xlabel('Days')
#plt.ylabel('Frequency')
#plt.show()

# Check for overlap between training and test sets
#overlap = set(X_train.index).intersection(set(X_test.index))
#print(f"Overlap between training and test sets: {len(overlap)} rows")


# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Cross-validation
#cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#print(f"Cross-Validation MAE: {-cv_scores.mean()}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", model.score(X_test, y_test))

# After training the next purchase prediction model
joblib.dump(model, '../models/next_purchase_model.pkl')

print("Next Purchase Prediction Model saved as 'next_purchase_model.pkl'")


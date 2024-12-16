# Goal: Predict if a customer will churn (i.e., stop engaging with the business) 
# based on their Recency, Frequency, and Monetary (RFM) metrics.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load the final dataset
final_data = pd.read_csv('../data/final_customer_segmented_data.csv')

#print(f"Number of duplicate rows: {final_data.duplicated().sum()}")
final_data = final_data.drop_duplicates()

# Create a churn label: 1 if Recency > 90 days (Churned), 0 if Recency <= 90 (Not Churned)
final_data['Churn'] = final_data['Recency'].apply(lambda x: 1 if x > 90 else 0)

# Select features (Recency, Frequency, Monetary, Cluster)
X = final_data[['Recency', 'Frequency', 'Monetary', 'Cluster']]
y = final_data['Churn']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model using cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred))

# Define the parameter grid
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of random samples
    scoring='accuracy',
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

# After training the churn prediction model
joblib.dump(model, '../models/churn_prediction_model.pkl')

print("Churn Prediction Model saved as 'churn_prediction_model.pkl'")



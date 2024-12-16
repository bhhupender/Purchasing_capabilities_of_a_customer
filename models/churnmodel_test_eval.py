import pandas as pd
import logging
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import silhouette_score

# Ensure the logs folder exists
log_folder = "../logs"
os.makedirs(log_folder, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_folder, 'churn_model_logs.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Load the trained model
    logging.info("Loading the trained churn prediction model...")
    model = joblib.load('../models/churn_prediction_model.pkl')

    # Load test data
    logging.info("Loading test data...")
    test_data = pd.read_csv('../data/churn_test_data.csv')

    # Separate features and target
    X_test = test_data.drop(columns=['Churn'])
    y_test = test_data['Churn']
    logging.info(f"Test data loaded with {X_test.shape[0]} records.")

    # Make predictions
    logging.info("Making predictions on test data...")
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    
    # Log evaluation metrics
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{report}")
    logging.info(f"AUC Score: {auc_score}")

    # Clustering evaluation using Silhouette Score
    try:
        logging.info("Evaluating clustering using Silhouette Score...")
        # Replace `model.labels_` with actual cluster labels
        silhouette_avg = silhouette_score(X_test, model.predict(X_test))  
        logging.info(f"Silhouette Score: {silhouette_avg}")
    except Exception as e:
        logging.error(f"Error during Silhouette Score calculation: {e}")

except Exception as e:
    logging.error(f"Error occurred: {e}")

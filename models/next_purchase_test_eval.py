import logging
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the logs folder exists
log_folder = "../logs"
os.makedirs(log_folder, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_folder, 'next_purchase_model_logs.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Load the trained model
    logging.info("Loading the trained next purchase prediction model...")
    model = joblib.load('../models/next_purchase_model.pkl')

    # Load test data
    logging.info("Loading test data...")
    test_data = pd.read_csv('../data/test_data_next_purchase.csv')

    # Separate features and target
    X_test = test_data.drop(columns=['TimeToNextPurchase'])
    y_test = test_data['TimeToNextPurchase']
    logging.info(f"Test data loaded with {X_test.shape[0]} records.")

    # Make predictions
    logging.info("Making predictions on test data...")
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log evaluation metrics
    logging.info(f"Mean Absolute Error (MAE): {mae}")
    logging.info(f"Mean Squared Error (MSE): {mse}")
    logging.info(f"R-squared Score: {r2}")

    # Log sample predictions
    logging.info("Logging sample predictions...")
    for i in range(min(10, len(predictions))):  # Log first 10 predictions
        logging.debug(f"Prediction {i+1}: {predictions[i]}, Actual: {y_test.iloc[i]}")

except Exception as e:
    logging.error(f"Error occurred: {e}")

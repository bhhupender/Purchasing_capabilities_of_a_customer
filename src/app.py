from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained models
churn_model = joblib.load('../models/churn_prediction_model.pkl')
next_purchase_model = joblib.load('../models/next_purchase_model.pkl')

@app.route('/')
def home():
    return "Customer Prediction API is running!"

# Endpoint for churn prediction
@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    # Get input data from POST request
    data = request.get_json()
    input_df = pd.DataFrame([data])
    
    # Predict churn (1 = Churned, 0 = Not Churned)
    churn_prediction = churn_model.predict(input_df)

    # Debugging print to check the response structure
    #print(f"Prediction: {churn_prediction[0]}")

    return jsonify({'Churn Prediction': int(churn_prediction[0])})

# Endpoint for next purchase prediction
@app.route('/predict_next_purchase', methods=['POST'])
def predict_next_purchase():
    # Get input data from POST request
    data = request.get_json()
    input_df = pd.DataFrame([data])
    
    # Predict time to next purchase
    next_purchase_prediction = next_purchase_model.predict(input_df)
    #print(f"Prediction: {next_purchase_prediction[0]}")
    return jsonify({'Next Purchase Prediction (days)': float(next_purchase_prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

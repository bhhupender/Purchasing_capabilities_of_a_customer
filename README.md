# Customer Prediction Application

## Overview
The **Customer Prediction Application** is a web-based solution that predicts:
1. **Customer Churn Likelihood**: Determines the probability of a customer leaving.
2. **Next Purchase Time**: Estimates the number of days until a customer's next purchase.

This application integrates a Flask API for backend machine learning model predictions and a Dash-based frontend dashboard for interactive user input and visualization.

---

## Features
- Predict customer churn using Random Forest Classifier.
- Estimate next purchase time using Random Forest Regressor.
- Interactive dashboard to visualize predictions.
- Comprehensive logging for model actions and outputs.

---

## System Architecture

![System Architecture](architecture_diagram.png)

---

## Prerequisites

### Software Requirements
- **Python 3.9+**
- Virtual Environment Tools (e.g., `venv` or `virtualenv`)
- AWS EC2 instance for deployment
- Nginx and Gunicorn for production setup

### Hardware Requirements
- AWS EC2 instance (t2.micro or higher)
- Local machine with SSH access

---

## Installation and Setup

### Clone the Repository
```bash
git clone <repository_url>
cd customer_prediction_app
```

### Set up the Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Environment Variables
Configure Flask app environment:
```bash
export FLASK_ENV=production
export FLASK_APP=src/app_new.py
```

### Run Locally
To start the application locally:
```bash
python src/app_new.py
```
Access the application at: `http://127.0.0.1:5000`

---

## Deployment

### Deploy on AWS EC2
1. **Launch an EC2 Instance**:
   - Use an Ubuntu or Amazon Linux AMI.
   - Allow port **5000** in the security group settings.

2. **Connect to the Instance**:
   ```bash
   ssh -i <key_file.pem> ec2-user@<instance_ip>
   ```

3. **Set Up the Environment**:
   - Install Python3, Pip, and virtual environment tools.
   - Clone the repository and install dependencies.

4. **Run Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 src.app_new:server
   ```

5. **Configure Nginx**:
   - Proxy requests from port **80** to Gunicorn.
   - Add HTTPS (optional).

---

## API Endpoints

### Churn Prediction
- **Endpoint**: `/predict_churn`
- **Method**: POST
- **Input**:
  ```json
  {
    "Recency": 10,
    "Frequency": 5,
    "Monetary": 5000,
    "Cluster": 1
  }
  ```
- **Output**:
  ```json
  {
    "Churn Prediction": 1
  }
  ```

### Next Purchase Prediction
- **Endpoint**: `/predict_next_purchase`
- **Method**: POST
- **Input**:
  ```json
  {
    "Recency": 10,
    "Frequency": 5,
    "Monetary": 5000,
    "Cluster": 1
  }
  ```
- **Output**:
  ```json
  {
    "Next Purchase Prediction (days)": 45.6
  }
  ```

---

## Dashboard Features
- **Inputs**:
  - Recency, Frequency, Monetary, and Cluster values.
- **Buttons**:
  - Predict Churn.
  - Predict Next Purchase.
- **Results**:
  - Displayed below the input fields.

---

## Logging
Logs are stored in the `/logs/` directory:
- **Churn Predictions**: `churn_prediction.log`
- **Next Purchase Predictions**: `next_purchase.log`

### Example Log Entry
```
INFO - 2024-12-16 14:30:12 - Input: {"Recency": 10, "Frequency": 5, "Monetary": 5000, "Cluster": 1}
INFO - 2024-12-16 14:30:12 - Output: {"Next Purchase Prediction (days)": 45.6}
```

---

## Model Details

### Churn Prediction Model
- **Algorithm**: Random Forest Classifier
- **Metrics**:
  - Accuracy: 98%

### Next Purchase Prediction Model
- **Algorithm**: Random Forest Regressor
- **Metrics**:
  - R² Score: 0.99

---

## Testing

### Testing the API
Use `test_data.csv` for API testing:
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"Recency": 10, "Frequency": 5, "Monetary": 5000, "Cluster": 1}' \
http://127.0.0.1:5000/predict_churn
```

### Unit Tests
Run unit tests for the models:
```bash
pytest tests/
```

---

## File Structure
```
customer_prediction_app/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── logs/                       # Log files
│   ├── churn_prediction.log
│   └── next_purchase.log
├── src/                        # Source code
│   ├── app_new.py              # Flask and Dash integrated application
│   ├── data_prep.py            # Data preprocessing script
│   ├── feature_engg.py         # Feature engineering script
│   └── app_dashboard.py        # Dash dashboard (if separated initially)
├── models/                     # Pre-trained ML models
│   ├── churn_prediction_model.pkl
│   └── next_purchase_model.pkl
├── data/                       # Sample and test data
│   ├── test_data.csv
│   └── test_data_next_purchase.csv
└── deployment/                 # Deployment scripts
    ├── gunicorn_start.sh
    └── nginx_config
```

---

## Future Enhancements
- Add user authentication for secure access.
- Enable real-time model retraining.
- Expand prediction features (e.g., customer segmentation).

---

## Contributors
- **Author**: Bhupender Singh
- **Contact**: bhhupender@gmail.com
- **GitHub**: https://github.com/bhhupender

---

## License
This project is licensed under the MIT License.

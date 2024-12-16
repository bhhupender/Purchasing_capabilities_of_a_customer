from flask import Flask, request, jsonify
import joblib
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import requests

# Initialize Flask app
server = Flask(__name__)

# Load trained models
#churn_model = joblib.load('../models/churn_prediction_model.pkl')
#next_purchase_model = joblib.load('../models/next_purchase_model.pkl')

churn_model = joblib.load('./models/churn_prediction_model.pkl')
next_purchase_model = joblib.load('./models/next_purchase_model.pkl')

# Flask Routes
@server.route('/')
def home():
    return "Integrated Customer Prediction Application is running!"

# Endpoint for churn prediction
@server.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    churn_prediction = churn_model.predict(input_df)
    return jsonify({'Churn Prediction': int(churn_prediction[0])})

# Endpoint for next purchase prediction
@server.route('/predict_next_purchase', methods=['POST'])
def predict_next_purchase():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    next_purchase_prediction = next_purchase_model.predict(input_df)
    return jsonify({'Next Purchase Prediction (days)': float(next_purchase_prediction[0])})

# Initialize Dash app
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Dash Layout
app.layout = html.Div([
    
    # Label and Input for Recency
    html.Div([
        html.Label("Bank Customer last visit(Days):"),
        dcc.Input(id='recency', type='number', placeholder='Enter Recency', style={'margin': '10px'})
    ], style={'margin-bottom': '15px'}),
    
    # Label and Input for Frequency
    html.Div([
        html.Label("Bank Customer visit frequency:"),
        dcc.Input(id='frequency', type='number', placeholder='Enter Frequency', style={'margin': '10px'})
    ], style={'margin-bottom': '15px'}),

    # Label and Input for Monetary
    html.Div([
        html.Label("Money in the account:"),
        dcc.Input(id='monetary', type='number', placeholder='Enter Monetary', style={'margin': '10px'})
    ], style={'margin-bottom': '15px'}),

    #dcc.Input(id='recency', type='number', placeholder='Enter Recency', style={'margin': '10px'}),
    #dcc.Input(id='frequency', type='number', placeholder='Enter Frequency', style={'margin': '10px'}),
    #dcc.Input(id='monetary', type='number', placeholder='Enter Monetary', style={'margin': '10px'}),
    #dcc.Dropdown(
    #    id='cluster',
    #    options=[{'label': str(i), 'value': i} for i in range(4)],
    #    placeholder='Select Cluster',
    #    style={'margin': '10px'}
    #),

    #Label and Dropdown for Cluster
    html.Div([
        html.Label("Customer Cluster:"),
        dcc.Dropdown(
            id='cluster',
            options=[{'label': f'Cluster {i}', 'value': i} for i in range(4)],  # Added descriptive labels for options
            placeholder='Select Cluster',
            style={'margin': '10px'}
        )
    ], style={'margin-bottom': '15px'}),

    html.Button('Predict Bank Account Closure', id='churn_button', n_clicks=0, style={'margin': '10px'}),
    html.Button('Predict Customer Next Purchase', id='purchase_button', n_clicks=0, style={'margin': '10px'}),
    html.Div(id='churn_output'),
    html.Div(id='purchase_output')
])

# Dash Callbacks
@app.callback(
    Output('churn_output', 'children'),
    Input('churn_button', 'n_clicks'),
    [Input('recency', 'value'),
     Input('frequency', 'value'),
     Input('monetary', 'value'),
     Input('cluster', 'value')]
)
def dash_predict_churn(n_clicks, recency, frequency, monetary, cluster):
    if n_clicks > 0:
        data = {
            'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 'Cluster': cluster
        }
        response = requests.post('http://127.0.0.1:5000/predict_churn', json=data)
        return f"Churn Prediction(Will customer close the account? if 0 then No, 1 yes): {response.json()['Churn Prediction']}"

@app.callback(
    Output('purchase_output', 'children'),
    Input('purchase_button', 'n_clicks'),
    [Input('recency', 'value'),
     Input('frequency', 'value'),
     Input('monetary', 'value'),
     Input('cluster', 'value')]
)
def dash_predict_next_purchase(n_clicks, recency, frequency, monetary, cluster):
    if n_clicks > 0:
        data = {
            'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 'Cluster': cluster
        }
        response = requests.post('http://127.0.0.1:5000/predict_next_purchase', json=data)
        return f"When customer will make Next Purchase(in days): {response.json()['Next Purchase Prediction (days)']}"

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=5000)
    #app.run(debug=True, host='0.0.0.0', port=5000)

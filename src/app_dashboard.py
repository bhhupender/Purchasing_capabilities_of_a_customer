import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Input(id='recency', type='number', placeholder='Enter Recency', style={'margin': '10px'}),
    dcc.Input(id='frequency', type='number', placeholder='Enter Frequency', style={'margin': '10px'}),
    dcc.Input(id='monetary', type='number', placeholder='Enter Monetary', style={'margin': '10px'}),
    dcc.Dropdown(
        id='cluster',
        options=[{'label': str(i), 'value': i} for i in range(4)],
        placeholder='Select Cluster',
        style={'margin': '10px'}
    ),
    html.Button('Predict Churn', id='churn_button', n_clicks=0, style={'margin': '10px'}),
    html.Button('Predict Next Purchase', id='purchase_button', n_clicks=0, style={'margin': '10px'}),
    html.Div(id='churn_output'),
    html.Div(id='purchase_output')
])

# Callback to predict churn
@app.callback(
    Output('churn_output', 'children'),
    Input('churn_button', 'n_clicks'),
    [Input('recency', 'value'),
     Input('frequency', 'value'),
     Input('monetary', 'value'),
     Input('cluster', 'value')]
)
def predict_churn(n_clicks, recency, frequency, monetary, cluster):
    if n_clicks > 0:
        data = {
            'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 'Cluster': cluster
        }
        response = requests.post('http://127.0.0.1:5000/predict_churn', json=data)
        return f"Churn Prediction: {response.json()['Churn Prediction']}"

# Callback to predict next purchase
@app.callback(
    Output('purchase_output', 'children'),
    Input('purchase_button', 'n_clicks'),
    [Input('recency', 'value'),
     Input('frequency', 'value'),
     Input('monetary', 'value'),
     Input('cluster', 'value')]
)
def predict_next_purchase(n_clicks, recency, frequency, monetary, cluster):
    if n_clicks > 0:
        data = {
            'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 'Cluster': cluster
        }
        response = requests.post('http://127.0.0.1:5000/predict_next_purchase', json=data)
        return f"Next Purchase Prediction (in days): {response.json()['Next Purchase Prediction (days)']}"

if __name__ == '__main__':
    app.run(debug=True, port=8050)

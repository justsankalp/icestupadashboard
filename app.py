import sqlite3
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --------------------------
# Autoencoder Model
# --------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --------------------------
# Preprocessing and Model Loading
# --------------------------

# Define feature names
features = ['water_pressure', 'log_ambient_temp', 'humidity', 'water_temp', 'ambient_temp']

# Load the scaler
scaler = StandardScaler()
scaler.mean_ = np.array([1.61401178, -7.11591429, 43.1693017, 1.10039844, -7.54093996])
scaler.scale_ = np.array([1.57396511, 4.15318914, 18.77094551, 0.68505492, 4.60274569])

# Load the trained Autoencoder model
model = Autoencoder(len(features))
model.load_state_dict(torch.load("autoencoder_model.pth"))
model.eval()

# Load precomputed training reconstruction errors
training_errors = np.load("training_errors.npy")
threshold = np.percentile(training_errors, 95)

# --------------------------
# Database Setup
# --------------------------

# Create or connect to SQLite database
conn = sqlite3.connect("realtime_data.db", check_same_thread=False)
cursor = conn.cursor()

# Create a table for storing real-time data
cursor.execute("""
CREATE TABLE IF NOT EXISTS realtime_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    water_pressure REAL,
    log_ambient_temp REAL,
    humidity REAL,
    water_temp REAL,
    ambient_temp REAL,
    valve_state TEXT,
    anomaly INTEGER
)
""")
conn.commit()

# --------------------------
# Evaluate Sample Function
# --------------------------

def evaluate_sample(sample_input):
    """
    Evaluate if a sample is an anomaly and identify feature contributions.
    """
    sample_df = pd.DataFrame([sample_input])
    sample_scaled = scaler.transform(sample_df[features].values)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstruction = model(sample_tensor)
        reconstruction_error = (sample_tensor - reconstruction).numpy().flatten()

    total_error = np.mean(reconstruction_error ** 2)
    anomaly = int(total_error > threshold)
    feature_contributions = dict(zip(features, reconstruction_error))

    return anomaly, feature_contributions

# --------------------------
# Dash Application
# --------------------------

# Function to clear the database table
def clear_database():
    """
    Clears the 'realtime_data' table in the database.
    """
    try:
        with sqlite3.connect("realtime_data.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM realtime_data")  # Deletes all records
            conn.commit()
            print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")

# --------------------------
# Dash Application
# --------------------------

app = Dash(__name__)
app.title = "Ice Stupa Viewer"

app.layout = html.Div(
    style={
        "backgroundColor": "#121212",
        "color": "#FFFFFF",
        "font-family": "Arial",
        "padding": "20px",
    },
    children=[
        html.H1(
            "Ice Stupa Real-Time Data Viewer",
            style={"textAlign": "center", "color": "#1DB954"}
        ),
        html.Div([
            html.Label("Select Valve States:", style={"color": "#1DB954"}),
            dcc.Dropdown(
                id="valve-state-dropdown",
                options=[{"label": "All", "value": "All"}],
                value="All",
                multi=False,
                style={"backgroundColor": "#333333", "color": "#FFFFFF"}
            ),
            html.Label("Select Features to Plot:", style={"color": "#1DB954", "marginTop": "10px"}),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": feature, "value": feature} for feature in features],
                value=["water_pressure"],
                multi=True,
                style={"backgroundColor": "#333333", "color": "#FFFFFF"}
            ),
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.H3("Real-Time Input", style={"color": "#1DB954"}),
            html.Div(id="real-time-values", style={"marginBottom": "20px"}),
            html.H3("Anomaly Detection Result", style={"color": "#1DB954"}),
            html.Div(id="output-result"),
            html.Div(id="output-feature-contributions", style={"whiteSpace": "pre-line"})
        ]),

        dcc.Graph(id="realtime-graph", style={"margin-top": "50px"}),

        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,  # Update every second
            n_intervals=0
        ),
    ]
)

# def max_contrib()

@app.callback(
    [Output("valve-state-dropdown", "options"),
     Output("real-time-values", "children"),
     Output("output-result", "children"),
     Output("output-feature-contributions", "children"),
     Output("realtime-graph", "figure")],
    [Input("interval-component", "n_intervals"),
     Input("feature-dropdown", "value"),
     Input("valve-state-dropdown", "value")]
)
def update_real_time_values(n_intervals, selected_features, selected_valve_state):
    try:
        # Fetch data from the server
        response = requests.get("http://127.0.0.1:5000/random-instance")  # Adjust the URL if necessary
        if response.status_code == 200:
            sample_input = response.json()
            sample_input["valve_state"] = "DRAIN"  # Example default

            # Evaluate anomaly
            anomaly, contributions = evaluate_sample(sample_input)

            # Insert current data into the database with the current timestamp
            current_timestamp = datetime.now().isoformat()
            with sqlite3.connect("realtime_data.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO realtime_data (timestamp, water_pressure, log_ambient_temp, humidity, water_temp, ambient_temp, valve_state, anomaly)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (current_timestamp, sample_input['water_pressure'], sample_input['log_ambient_temp'], 
                      sample_input['humidity'], sample_input['water_temp'], sample_input['ambient_temp'], 
                      sample_input['valve_state'], anomaly))
                conn.commit()

            # Fetch accumulated data
            query = "SELECT * FROM realtime_data"
            if selected_valve_state != "All":
                query += f" WHERE valve_state='{selected_valve_state}'"
            query += " ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn)

        # Ensure DataFrame isn't empty
        if df.empty:
            return [], "No data available.", "No data available.", "", {}

        # Prepare graph
        fig = px.line(
            df,
            x="timestamp",
            y=selected_features,
            title="Real-Time Data Plot",
            labels={"timestamp": "Timestamp"},
        )
        fig.update_layout(template="plotly_dark")

        # Update dropdown options with non-null valve states
        valve_states = [{"label": "All", "value": "All"}] + [{"label": state, "value": state} for state in df["valve_state"].dropna().unique()]

        return (
            valve_states,
            f"Real-Time Data:\n{sample_input}",
            "The sample is an anomaly." if anomaly else "The sample is normal.",
            "\n".join([f"{k}: {v:.4f}" for k, v in contributions.items()]),
            fig
        )

    except Exception as e:
        print(f"Error: {e}")
        return [], "Error fetching data.", "", "", {}

if __name__ == "__main__":
    clear_database()  # Clear the database before starting the app
    app.run(debug=True)

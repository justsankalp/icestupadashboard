from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('/home/sankalpkarthi/Documents/Ice Stupa/game/dash/merge.csv')  # Replace with the actual path to your CSV file

# Global variable to keep track of the current row
current_index = 0

@app.route('/random-instance', methods=['GET'])
@app.route('/random-instance', methods=['GET'])
def get_next_instance():
    global current_index
    print(f"Serving row: {current_index}")  # Log the current row
    instance = df.iloc[current_index].to_dict()
    current_index = (current_index + 1) % len(df)
    return jsonify({
        'water_pressure': instance['water_pressure'],
        'log_ambient_temp': instance['log_ambient_temp'],
        'humidity': instance['humidity'],
        'water_temp': instance['water_temp'],
        'ambient_temp': instance['ambient_temp']
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Server runs on port 5000

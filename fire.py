import requests
import time
from datetime import datetime

# Ollama API URL (Replace with the actual endpoint)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to call Ollama API
import json
import html
import requests

def call_ollama(sensor_data):
    prompt_template = """
    Using the following weather station readings:
    - Pressure: {pressure} Pa
    - Ambient Temperature: {ambient_temp} °C
    - Humidity: {humidity} %
    - Ground Temperature: {ground_temp} °C
    - UV Intensity: {uv_intensity} mW/cm²
    - Wind Direction: {wind_direction}° ({wind_direction_cardinal})
    - Wind Speed: {wind_speed} m/s
    - Rain & Snow: {rain_snow},

    Please generate a fun and engaging response that includes:
    1. A short description of the current weather conditions in Ladakh based on the data.
    2. An interesting or fun fact about Ladakh, its geography, culture, or climate, related to the weather data.
    3. A concluding sentence that encourages curiosity about Ladakh's unique environment.
    """
    prompt = prompt_template.format(
        pressure=sensor_data['pressure'],
        ambient_temp=sensor_data['ambient_temp'],
        humidity=sensor_data['humidity'],
        ground_temp=sensor_data['ground_temp'],
        uv_intensity=sensor_data['uv_intensity'],
        wind_direction=sensor_data['wind_direction'],
        wind_direction_cardinal=sensor_data.get('wind_direction_cardinal', "Unknown"),
        wind_speed=sensor_data['wind_speed'],
        rain_snow=sensor_data['rain_snow']
    )

    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Unescape and extract the meaningful response
        unescaped_response = html.unescape(data.get("response", "No response provided."))
        return unescaped_response

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except json.JSONDecodeError as e:
        return f"JSON decode failed: {e}. Raw response: {response.text}"

# Function to fetch sensor data and call the API
def process_data():
    # Fetch current sensor data (replace with your actual data source)
    sensor_data = {
        "pressure": 63567.55,
        "ambient_temp": -0.00,
        "humidity": 39.80,
        "ground_temp": -3.56,
        "uv_intensity": 467.27,
        "wind_direction": 68,
        "wind_direction_cardinal": "East",
        "wind_speed": 1.23,
        "rain_snow": "No"
    }

    # Call the Ollama API
    response = call_ollama(sensor_data)

    # Log or display the response
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{timestamp}] Generated Response:\n{response}")
    resp_edit = response.split("</think>")[1]
    resp_out = resp_edit.split("\n")
    x = ""
    for i in resp_out:
        if len(i) > 30:
            x+=i
            x+="\n"
    out = ""
    for i in x:
        if i!="*" and i!="#":
            out+=i

    return out

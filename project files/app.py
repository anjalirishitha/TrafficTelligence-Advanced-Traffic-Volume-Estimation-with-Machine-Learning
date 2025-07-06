import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract values from form
        holiday = int(request.form.get('holiday'))
        temp = float(request.form.get('temp'))
        rain = float(request.form.get('rain'))
        snow = float(request.form.get('snow'))
        weather = int(request.form.get('weather'))

        # Get date and time
        date_str = request.form.get('date')      # Format: YYYY-MM-DD
        time_str = request.form.get('Time')      # Format: HH:MM

        # Combine and parse datetime
        datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        year = datetime_obj.year
        month = datetime_obj.month
        day = datetime_obj.day
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        second = 0

        # Prepare input for prediction
        input_data = np.array([[holiday, temp, rain, snow, weather,
                                year, month, day, hour, minute, second]])
        
        input_scaled = scale.transform(input_data)
        prediction = model.predict(input_scaled)

        prediction_text = f"🚗 Estimated Traffic Volume is: {int(prediction[0])} vehicles"

        # Show result on result.html instead of index.html
        return render_template("result.html", prediction_text=prediction_text)

    except Exception as e:
        return render_template("result.html", prediction_text=f"❌ Error: {e}")

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

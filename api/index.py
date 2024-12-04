from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import heartpy as hp

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('blood_pressure_and_respiratory_prediction_model.h5')

# Define prediction route
@app.route('/', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        _, m = hp.process(np.array(data["ppg"]), sample_rate = 100.0)

        data.pop("ppg")
        m.pop("pnn20")
        m.pop("pnn50")
        m.pop("hr_mad")
        m.pop("sd1")
        m.pop("sd2")
        m.pop("s")
        m.pop("sd1/sd2")
        m.pop("breathingrate")

        input_data = {**data, **m}

        # Extract and preprocess input features
        input_features = np.array([
          input_data['bpm'],
          input_data['ibi'],
          input_data['sdnn'],
          input_data['sdsd'],
          input_data['rmssd'],
          input_data['age'],
          input_data['weight']
        ]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(input_features)
        systole, diastole, respiratory_rate = prediction[0]

        # Return predictions as JSON
        return jsonify({
            'status': 'success',
            'message': 'Data berhasil diproses',
            'estimated_systole': float(systole),
            'estimated_diastole': float(diastole),
            'estimated_respiratory_rate': float(respiratory_rate),
            **m
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

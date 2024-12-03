from flask import Flask, request, jsonify
import joblib
import pandas as pd
import heartpy as hp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('blood_pressure_and_respiratory_prediction_model.h5')
#systole_model = joblib.load('systole_model.pkl')
#diastole_model = joblib.load('diastole_model.pkl')

# Load scaler parameters (you need to save this during your training process and load here)
data = pd.read_csv('data-extraction-r-normalized.csv')
scaler = StandardScaler()
scaler.fit(data[['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'age', 'weight']])

@app.route('/api/predict', methods=['POST'])
def predict():
  try:
    input_data = request.get_json()
    
    # Validate input
    required_keys = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'age', 'weight']
    if not all(key in input_data for key in required_keys):
        return jsonify({'error': f'Missing required keys. Required keys: {required_keys}'}), 400

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
    input_scaled = scaler.transform(input_features)

    # Make predictions
    prediction = model.predict(input_scaled)
    systole, diastole, respiratory_rate = prediction[0]

    # Return predictions as JSON
    return jsonify({
        'predicted_systole': float(systole),
        'predicted_diastole': float(diastole),
        'predicted_respiratory_rate': float(respiratory_rate)
    })
    
  except Exception as e:
    return jsonify({'status': 'error','message': str(e)}), 500
    
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

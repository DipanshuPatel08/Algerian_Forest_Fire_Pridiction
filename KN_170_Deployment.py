import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

KN_170_Deployment = Flask(__name__)
app = KN_170_Deployment

# Load Ridge Regression Model and StandardScaler
try:
    reg_model = pickle.load(open('KN_159_Ridge_Reg.pkl', 'rb'))
    scaler = pickle.load(open('KN_159_standardScaler.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)  # Stop execution if files are missing

@app.route('/')
def index():
    """ Home Page """
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    """ Prediction Page and Model Inference """
    if request.method == 'POST':
        try:
            # Extract form data
            features = [float(request.form[key]) for key in ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI']]
            
            # Add default values for Classes and Region
            features.extend([0, 0])  # Replace 0 with appropriate default values for Classes and Region
            
            input_data = pd.DataFrame([features], columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])
            
            scaled_data = scaler.transform(input_data)
            
            prediction = reg_model.predict(scaled_data)[0]

            return render_template('home.html', result=round(prediction, 2))

        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
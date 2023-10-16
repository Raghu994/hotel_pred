from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import gzip

app = Flask(__name__)
CORS(app) 

# Load the trained model
#model = joblib.load('hotel_cancellation_model.pkl')
with gzip.open('hotel_cancellation_model.pkl.gz', 'rb') as f:
    loaded_model = joblib.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the HTML form
        #data = request.get_json()
        #print("data",  data)
        # Create a DataFrame from the input data
        hotel = int(request.form['hotel'])
        lead_time = request.form['lead_time']
        arrival_date_week_number = request.form['arrival_date_week_number']
        adults =request.form['adults']
        children= request.form['children']
        babies =request.form['babies']
        days_in_waiting_list = request.form['days_in_waiting_list']
        adr = request.form['adr']
        required_car_parking_spaces = request.form['required_car_parking_spaces']
        total_of_special_requests = request.form['total_of_special_requests']
        meal = int(request.form['meal'])
        distribution_channel = int(request.form['distribution_channel'])
        reserved_room_type = int(request.form['reserved_room_type'])
        deposit_type = int(request.form['deposit_type'])
        input_data = np.array([[hotel, lead_time, arrival_date_week_number, adults, children, babies, days_in_waiting_list, adr, required_car_parking_spaces, total_of_special_requests, meal, distribution_channel, reserved_room_type, deposit_type]])
        #{'hotel': '1', 'lead_time': '50', 'arrival_date_week_number': '27', 'adults': '2', 'children': '1', 'babies': '0', 'days_in_waiting_list': '0', 'adr': '105.5', 'required_car_parking_spaces': '1', 'total_of_special_requests': '2', 'meal': 'BB', 'distribution_channel': 'Corporate', 'reserved_room_type': 'A', 'deposit_type': 'No Deposit'}
        print(input_data)
        #print("abrakdabra")
        # Make predictions and get probability of cancellation (class 1)
        probabilities = loaded_model.predict_proba(input_data)[:, 1]        
        # Apply a threshold (e.g., 0.5) to determine the label
        predictions = ['Cancelled' if prob >= 0.5 else 'Not Cancelled' for prob in probabilities]
        
        return render_template('index.html', probabilities=probabilities, predictions=predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return a 400 Bad Request status code for errors

if __name__ == '__main__':
    app.run(debug=True)

# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app
superkart_api = Flask("SuperKart")

# Load the trained model pipeline
model = joblib.load("superkart_model.joblib")

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System - Preprocessed Prediction API"

@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Get JSON data from the request payload
        data = request.get_json()

        # Safely capture variables directly as configured/molded
        sample = {
            'Product_Weight': data.get('Product_Weight', 12.66),
            'Product_Sugar_Content': data.get('Product_Sugar_Content', 'Low Sugar'),
            'Product_Allocated_Area': data.get('Product_Allocated_Area', 0.027),
            'Product_MRP': data.get('Product_MRP', 117.08),
            'Store_Id': data.get('Store_Id', 'OUT004'),
            'Store_Size': data.get('Store_Size', 'Medium'),
            'Store_Location_City_Type': data.get('Store_Location_City_Type', 'Tier 2'),
            'Store_Type': data.get('Store_Type', 'Supermarket Type2'),
            'Product_Id_char': data.get('Product_Id_char', 'FD'),
            'Product_Category': data.get('Product_Category', 'Food'),
            'Store_Age_Years': data.get('Store_Age_Years', 16)
        }

        # Convert the sample into a DataFrame
        input_data = pd.DataFrame([sample])

        # Align order of columns with pipeline expectation
        expected_columns = [
            'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_MRP', 'Store_Id', 'Store_Size', 'Store_Location_City_Type',
            'Store_Type', 'Product_Id_char', 'Product_Category', 'Store_Age_Years'
        ]
        input_data = input_data[expected_columns]

        # Generate prediction using the trained model
        prediction = model.predict(input_data).tolist()[0]

        return jsonify({'Sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 400

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        file = request.files['file']
        input_data = pd.read_csv(file)

        # Ensure the batch has all expected columns
        expected_columns = [
            'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_MRP', 'Store_Id', 'Store_Size', 'Store_Location_City_Type',
            'Store_Type', 'Product_Id_char', 'Product_Category', 'Store_Age_Years'
        ]
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = np.nan

        input_data = input_data[expected_columns]
        predictions = model.predict(input_data).tolist()

        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': f"Batch prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    superkart_api.run(debug=True)

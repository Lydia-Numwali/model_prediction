from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import joblib
import json

# Load the model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Load the dataset to get the original feature names and values
df = pd.read_csv('student_depression_dataset.csv')

# Get feature names from the dataset (excluding target columns and ID)
target_columns = ['Depression Level', 'Anxiety Level', 'Stress Level']
feature_names = [col for col in df.columns if col not in target_columns]
print("Model feature names:", feature_names)  # Debug log

# Define feature values for the frontend
feature_values = {
    'id': list(range(1, 1001)),  # Example range
    'Gender': ['Male', 'Female'],
    'City': sorted(df['City'].unique().tolist()),
    'Profession': sorted(df['Profession'].unique().tolist()),
    'Work Pressure': ['Low', 'Medium', 'High'],
    'CGPA': [round(x * 0.1, 1) for x in range(20, 41)],  # 2.0 to 4.0
    'Study Satisfaction': ['Low', 'Medium', 'High'],
    'Job Satisfaction': ['Low', 'Medium', 'High'],
    'Sleep Duration': list(range(4, 13)),  # 4 to 12 hours
    'Dietary Habits': ['Poor', 'Fair', 'Good'],
    'Degree': sorted(df['Degree'].unique().tolist()),
    'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
    'Work/Study Hours': list(range(1, 17)),  # 1 to 16 hours
    'Financial Stress': ['Low', 'Medium', 'High'],
    'Family History of Mental Illness': ['Yes', 'No'],
    'Depression': ['Yes', 'No']
}

def home(request):
    return render(request, 'prediction_app/home.html', {
        'feature_names': feature_names,
        'feature_values': feature_values
    })

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Parse JSON data
            data = json.loads(request.body)
            print("Received data:", data)
            
            # Validate input data
            if not all(field in data for field in feature_names):
                missing_fields = [field for field in feature_names if field not in data]
                return JsonResponse({
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }, status=400)
            
            # Process input data
            try:
                input_data = {}
                for feature in feature_names:
                    if feature == 'id':
                        input_data[feature] = int(data[feature])
                    elif feature == 'CGPA':
                        input_data[feature] = float(data[feature])
                    elif feature == 'Sleep Duration' or feature == 'Work/Study Hours':
                        input_data[feature] = int(data[feature])
                    elif feature in ['Depression', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
                        input_data[feature] = 1 if data[feature] == 'Yes' else 0
                    else:
                        # For categorical features, use the index in the feature_values list
                        input_data[feature] = feature_values[feature].index(data[feature])
            except (ValueError, KeyError, IndexError) as e:
                return JsonResponse({
                    'error': f'Invalid input values: {str(e)}'
                }, status=400)
            
            print("Processed input data:", input_data)
            
            # Create DataFrame with feature names
            X = pd.DataFrame([input_data], columns=feature_names)
            
            # Scale features
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                print("Scaling error:", str(e))
                return JsonResponse({
                    'error': 'Error scaling input data'
                }, status=500)
            
            # Make prediction
            try:
                predictions = model.predict(X_scaled)
            except Exception as e:
                print("Prediction error:", str(e))
                return JsonResponse({
                    'error': 'Error making prediction'
                }, status=500)
            
            print("Raw predictions:", predictions)
            
            # Format response
            response = {
                'predictions': predictions[0].tolist(),
                'feature_names': ['Depression Level', 'Anxiety Level', 'Stress Level']
            }
            
            print("Sending response:", response)
            return JsonResponse(response)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            print("Unexpected error:", str(e))
            return JsonResponse({
                'error': f'Server error: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'error': 'Invalid request method'
    }, status=405)

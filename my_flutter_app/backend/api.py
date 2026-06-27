
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import datetime
from sklearn.preprocessing import StandardScaler
import uvicorn

# Create FastAPI instance
app = FastAPI()

# Test route to ensure FastAPI is working
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/test")
def test_route():
    return {"message": "FastAPI app is working!"}

# Load the trained LSTM model
try:
    model = tf.keras.models.load_model('backend/finall.keras', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define a request model for prediction
class PeriodPredictionRequest(BaseModel):
    cycle_lengths: list[int]

# Define the route for period prediction
@app.post("/predict_period")
def predict_period(request: PeriodPredictionRequest):
    cycle_lengths = request.cycle_lengths

    # Check if cycle_lengths exists and handle different formats
    if len(cycle_lengths) != 5:
        raise HTTPException(status_code=400, detail="Please provide exactly five cycle lengths.")
    
    try:
        # Create second feature as the average cycle length
        avg_period = np.mean(cycle_lengths)
        
        # Prepare input data: Cycle lengths and average cycle length as features
        features = np.array([[cycle_length, avg_period] for cycle_length in cycle_lengths])

        # Initialize and fit the scalers with the features (fit them dynamically)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Fit the scalers based on the input data
        scaler_X.fit(features)
        scaler_y.fit(np.array(cycle_lengths).reshape(-1, 1))

        # Standardize the input data using the newly fitted scalers
        features_scaled = scaler_X.transform(features)
        features_scaled = features_scaled.reshape(1, 5, 2)  # Reshape to (1, 5, 2) for LSTM model
        
        # Make the prediction using the model
        predicted_next_period_scaled = model.predict(features_scaled)
        
        # Inverse transform the prediction to get the actual cycle length
        predicted_next_period = scaler_y.inverse_transform(predicted_next_period_scaled.reshape(-1, 1))
        
        # Round off the predicted cycle length
        predicted_next_period_days = round(float(predicted_next_period[0][0]))
        
        # Calculate the predicted next period date
        today = datetime.datetime.today()
        predicted_next_period_date = today + datetime.timedelta(days=predicted_next_period_days)
        
        # Calculate the ovulation date (14 days before the predicted next period)
        ovulation_date = predicted_next_period_date - datetime.timedelta(days=14)
        
        # Return the results
        return {
            'predicted_next_period_date': predicted_next_period_date.strftime('%Y-%m-%d'),
            'predicted_ovulation_date': ovulation_date.strftime('%Y-%m-%d'),
            'predicted_next_period_cycle_length': predicted_next_period_days  # Rounded value
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app on port 8001
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)

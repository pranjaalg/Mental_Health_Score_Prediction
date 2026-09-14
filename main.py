import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load trained pipeline/model
model = joblib.load('Mental_Health_Model.pkl')
top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Serve index.html at root if hosting frontend together on Render
@app.get("/")
def read_root():
    return FileResponse("index.html")

# Input Schema matching script.js
class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal['Male', 'Female', 'Other']
    country: str
    academic_level: Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform: Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal['Medium', 'Low', 'Very High', 'High']


# Output Schema matching script.js expectation
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    try:
        # Map country to top_countries or fallback to 'Other'
        country_group = data.country if data.country in top_countries else "Other"

        # Construct exact DataFrame input format expected by ColumnTransformer / Pipeline
        input_data = {
            'Age': data.age,
            'Gender': data.gender,
            'Country': data.country,
            'Academic_Level': data.academic_level,
            'Most_Used_Platform': data.most_used_platform,
            'Purpose_Of_Use': data.purpose_of_use,
            'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
            'Daily_Unlocks': data.daily_unlocks,
            'Study_Hours': data.study_hours,
            'Physical_Activity_Hours': data.physical_activity_hours,
            'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
            'Stress_Level': data.stress_level,
            'Grouped_Country': country_group
        }

        input_df = pd.DataFrame([input_data])

        # Generate model prediction
        prediction = model.predict(input_df)[0]
        
        # Return formatted floating score rounded to 2 decimal places
        return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
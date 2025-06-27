from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://beamxsolutions.netlify.app",
        "http://localhost:3000"  # For local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the model and feature columns
model = joblib.load("loan_approval_simple_rf.pkl")
feature_columns = joblib.load("simple_feature_columns.pkl")

# Define input data model using Pydantic
class LoanApplication(BaseModel):
    Dependents: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender_Male: int
    Married_Yes: int
    Education_Not_Graduate: int
    Self_Employed_Yes: int
    Property_Area_Semiurban: int
    Property_Area_Urban: int

@app.get("/")
def read_root():
    return {"message": "Loan Approval API"}

@app.post("/predict")
async def predict_loan_approval(application: LoanApplication):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.dict()], columns=feature_columns)
        
        # Make prediction
        prob = model.predict_proba(input_data)[0, 1]
        prediction = model.predict(input_data)[0]
        
        return {
            "approval_probability": float(prob),
            "prediction": "APPROVED" if prediction else "REJECTED"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
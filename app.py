# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load artifacts
model   = joblib.load('loan_approval_rf.pkl')
columns = joblib.load('feature_columns.pkl')

# Define request schema
class Applicant(BaseModel):
    Gender_Male: int
    Married_Yes: int
    Education_NotGraduate: int
    Self_Employed_Yes: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area_Semiurban: int
    Property_Area_Urban: int

app = FastAPI()

# Set cutoff threshold for approval
cutoff = 0.5  # You can adjust this value based on your ROC analysis

@app.post('/predict')
def predict(applicant: Applicant):
    data = pd.DataFrame([applicant.dict()], columns=columns)
    prob = model.predict_proba(data)[0,1]
    approved = int(prob >= cutoff)  # use your tuned cutoff
    return {'approval_probability': prob, 'approved': approved}

# Run with: uvicorn app:app --reload
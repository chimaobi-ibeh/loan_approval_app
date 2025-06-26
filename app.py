from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('loan_approval_rf.pkl')
columns = joblib.load('feature_columns.pkl')

# Define request schema with validation
class Applicant(BaseModel):
    Gender_Male: int = Field(ge=0, le=1)
    Married_Yes: int = Field(ge=0, le=1)
    Education_NotGraduate: int = Field(ge=0, le=1)
    Self_Employed_Yes: int = Field(ge=0, le=1)
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: float = Field(ge=0)
    Loan_Amount_Term: float = Field(ge=0)
    Credit_History: int = Field(ge=0, le=1)
    Property_Area_Semiurban: int = Field(ge=0, le=1)
    Property_Area_Urban: int = Field(ge=0, le=1)

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

@app.get("/")
async def root():
    return {"message": "Loan Approval API. Use /predict for predictions. Docs at /docs."}

@app.post('/predict')
async def predict(applicant: Applicant):
    data = pd.DataFrame([applicant.dict()], columns=columns)
    prob = model.predict_proba(data)[0,1]
    approved = int(prob >= 0.5)
    return {'approval_probability': prob, 'approved': approved}
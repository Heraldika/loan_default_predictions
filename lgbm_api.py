import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

model = joblib.load('./lgbm_rfec_model.dump')
    
class ResultData(BaseModel):
    loan_default_pred: int
    loan_default_proba: float
    
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
           <html>
               <body>
                   <h1>Welcome to the loan default prediction API :)</h1>
                   <p>To get data on loan POST loan info to /test</p>
               </body>
           </html>
           """

@app.post('/test', response_model=ResultData)
async def test_input(data: Request):
    X = await data.json()
    data = pd.read_json(X, orient='records', typ='series').values
    default = model.predict([data])
    proba = model.predict_proba([data])[:, 1]
    return {'loan_default_pred': default, 'loan_default_proba': proba}




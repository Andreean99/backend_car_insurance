from flask import Flask, jsonify, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)


LABEL = ['Not Claim Loan', 'Claim Loan']
columns = ['EDUCATION', 'INCOME', 'CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'DRIVING_EXPERIENCE', 'VEHICLE_OWNERSHIP', 'MARRIED', 'CHILDREN']
with open("asuransi.pkl", "rb") as f:
    model_insurance = pickle.load(f)

@app.route("/")
def homepage():
    return "<h1>Backend Pemodelan Car Insurance </h1>"

@app.route("/insurance", methods=["GET", "POST"])
def insurance_inference():
    if request.method == 'POST':
        data = request.json
        print(data)
        new_data = [data['EDUCATION'],
                    data['INCOME'],
                    data['CREDIT_SCORE'],
                    data['ANNUAL_MILEAGE'],
                    data['SPEEDING_VIOLATIONS'],
                    data['PAST_ACCIDENTS'],
                    data['DRIVING_EXPERIENCE'],
                    data['VEHICLE_OWNERSHIP'],
                    data['MARRIED'],
                    data['CHILDREN']]

        new_data = pd.DataFrame([new_data],columns=columns)

        res = model_insurance.predict(new_data)
        print("res :", res )
        response = {'code':200, 'status':'OK',
                    'result':{'prediction': str(res[0]),
                              'classes': LABEL[int(res[0])]}}


        return jsonify(response)
    return "Silahkan gunakan method post untuk mengakses model insurance"
# app.run(debug=True)
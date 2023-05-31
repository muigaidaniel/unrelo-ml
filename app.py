import pickle
from flask import Flask, request, jsonify, render_template
from datetime import datetime,timedelta, date
import numpy as np
import pandas as pd
from predict_pipeline import create_features, generate_hourly_data, predict_data

app = Flask(__name__)
model = pickle.load(open('reg2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method =='GET':
        return render_template('index.html')
    if request.method =='POST':
        json_data = request.get_json(force=True)
        
        # start_date = request.form["start_date"]
        # end_date = request.form["end_date"]
        sensor_id = json_data['sensor_id']
        start_date = json_data['start_date']
        end_date = json_data["end_date"]
        df = generate_hourly_data(start_date, end_date)
        df=create_features(df)
        prediction_df = predict_data(model,df,sensor_id)
        json_data = prediction_df.to_json(orient='index', date_format='iso')
        return json_data
        # return render_template('prediction.html', prediction=json_data)
        
       
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=False)
        



        






from predict_pipeline import create_features, generate_hourly_data, predict_data
import pickle
import pandas as pd

model = pickle.load(open('reg.pkl', 'rb'))
hourly_df=generate_hourly_data('2023-05-20','2023-05-22')
hourly_df=create_features(hourly_df)
pred_df=predict_data(model,hourly_df)
json_data = pred_df.to_json(orient='records')
print(pred_df)
print(json_data)

import pandas as pd

def generate_hourly_data(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    hourly_dates = pd.date_range(start=start, end=end, freq='H')
    df = pd.DataFrame( columns=['avg_temp', 'avg_humidity', 'timestamp (UTC)'])
    df['timestamp (UTC)'] = hourly_dates
    return df

hourly_df = generate_hourly_data('2023-05-20', '2023-05-22')


def create_features(df):
    df = df.copy()
    df['timestamp (UTC)'] = pd.to_datetime(df['timestamp (UTC)'])
    df['hour'] = df['timestamp (UTC)'].dt.hour
    df['dayofweek'] = df['timestamp (UTC)'].dt.dayofweek
    df['quarter'] = df['timestamp (UTC)'].dt.quarter
    df['month'] = df['timestamp (UTC)'].dt.month
    df['year'] = df['timestamp (UTC)'].dt.year
    df['dayofyear'] = df['timestamp (UTC)'].dt.dayofyear
    return df

def predict_data(model,df,sensor_id):
    df['sensor_id'] = sensor_id
    
    features=['dayofweek','quarter','month','year','dayofyear','hour','sensor_id']
    prediction = model.predict(df[features])
    prediction_df = pd.DataFrame(index=df.index)
    prediction_df['avg_temp']=prediction[:,0]
    prediction_df['avg_humidity']=prediction[:,1]
    # prediction_df['daily_avg_temp']=prediction_df['avg_temp'].rolling(24).mean()
    # prediction_df['daily_avg_humidity']=prediction_df['avg_humidity'].rolling(24).mean()
    # json_data = prediction_df.to_json(orient='records')
    return prediction_df


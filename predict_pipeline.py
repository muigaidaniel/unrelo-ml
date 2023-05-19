import pandas as pd

def generate_hourly_data(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    hourly_dates = pd.date_range(start=start, end=end, freq='H')
    df = pd.DataFrame(index=hourly_dates, columns=['avg_temp', 'avg_humidity'])
    df.index.name = 'timestamp (UTC)'
    df.index=pd.to_datetime(df.index)
    return df

def create_features(df):
    df=df.copy()
    df['hour']=df.index.hour
    df['dayofweek']=df.index.dayofweek
    df['quarter']=df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year
    df['dayofyear']=df.index.dayofyear
    return df

def predict_data(model,df):
    features=['dayofweek','quarter','month','year','dayofyear','hour']
    prediction = model.predict(df[features])
    prediction_df = pd.DataFrame(index=df.index)
    prediction_df['avg_temp']=prediction[:,0]
    prediction_df['avg_humidity']=prediction[:,1]
    # json_data = prediction_df.to_json(orient='records')
    return prediction_df


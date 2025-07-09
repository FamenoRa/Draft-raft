 # loading & cleaning the data
import os
import pandas as pd

DATA_PATH = os.getenv('TRAFFIC_DATA_PATH','data/traffic_datas.csv')

def load_and_clean(path: str) -> pd.DataFrame:
    """Load traffic data CSV and clean it."""
    df = pd.read_csv(path, parse_dates=['DateTime'], infer_datetime_format=True)
    df = df.rename(columns={
        'DateTime': 'datetime',
        'Junction': 'junction_id',
        'Vehicles': 'vehicle_count'
    }).drop(columns=['ID'], errors='ignore')
    df['vehicle_count'] = df['vehicle_count'].clip(lower=0)
    df = df.drop_duplicates(['junction_id', 'datetime']).set_index('datetime')

    # add hour feature
    df['hour'] = df.index.hour
    return df

def group_by_junction_hour(df: pd.DataFrame):
    d = df.groupby(['junction_id', 'hour'])['vehicle_count'].median().reset_index()
    return d

def get_data():
    dt = load_and_clean(DATA_PATH)
    dt.to_csv('data/cleaned_data.csv')
    return dt
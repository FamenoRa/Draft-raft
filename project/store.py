import os
import pandas as pd

DATA_PATH = os.getenv('TRAFFIC_DATA_PATH','data/traffic_datas.csv')

def store_data(d, n):
    df = pd.DataFrame(d)
    df.to_csv(f'data/{n}', index=False)

def retrieve_data(n):
    return pd.read_csv(f'data/{n}')

def retrieve_hour_vehicles(d, h, j):
    return d[(d['hour'] == h) & (d['junction_id'] == j)]

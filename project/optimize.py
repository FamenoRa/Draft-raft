import os
from store import store_data, retrieve_data
from QAOA_algo import optimize_light_cycle
from data_clean import load_and_clean
from data_training import train_congestion_model
from QAOA_algo import optimize_light_cycle

DATA_PATH = os.getenv('TRAFFIC_DATA_PATH','data/traffic_datas.csv')
def get_data():
    return load_and_clean(DATA_PATH)

def get_model(df):
    return train_congestion_model(df)

if __name__ == "__main__":
    hour = 10
    vehicles = 20
    ratio = 0.5
    cycle = 50
    df = get_data()
    df.to_csv('data/cleaned_data.csv')
    model = get_model(df)
    res = optimize_light_cycle(vehicles, ratio, cycle)
    store_data([res], 'results.csv')
    d = retrieve_data('results.csv')
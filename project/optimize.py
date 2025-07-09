import pandas as pd
from store import store_data, retrieve_data
from QAOA_algo import optimize_light_cycle
from data_clean import get_data, group_by_junction_hour
from data_training import train_congestion_model
from QAOA_algo import optimize_light_cycle

def get_model(df):
    return train_congestion_model(df)

if __name__ == "__main__":
    ratio = 0.5
    cycle = 50
    df = get_data()
    d = group_by_junction_hour(df)
    # model = get_model(df)
    res = d['vehicle_count'].apply(lambda v: optimize_light_cycle(v, ratio, cycle)).apply(pd.Series)
    d = pd.concat([d, res],axis=1)
    print(d.head())
    store_data(d, 'results.csv')
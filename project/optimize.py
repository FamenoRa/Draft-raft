import time
import swifter
import pandas as pd
from store import store_data, retrieve_data
from QAOA_algo import optimize_light_cycle
from data_clean import get_data, group_by_junction_hour
from data_training import train_congestion_model
from QAOA_algo import optimize_light_cycle

def get_model(df):
    return train_congestion_model(df)

if __name__ == "__main__":
    start_time = time.time()
    ratio = 0.7
    cycle = 60
    df = get_data()
    d = group_by_junction_hour(df)
    # model = get_model(df)
    res = d['vehicle_count'].swifter.apply(lambda v: optimize_light_cycle(v, ratio, cycle)).apply(pd.Series)
    d = pd.concat([d, res],axis=1)
    store_data(d, 'results.csv')
    end_time = time.time()
    print(f"Execution took {end_time - start_time:.2f} seconds.")
import time
import swifter
import pandas as pd
from store import store_data, retrieve_data
from data_clean import get_data, group_by_junction_hour, group_by_junction
from data_training import train_congestion_model
from QAOA_algo import optimize_light_cycle

def get_model(df):
    return train_congestion_model(df)

if __name__ == "__main__":
    start_time = time.time()
    cycle = 60
    df = get_data()
    d = group_by_junction_hour(df)
    hours = range(1)
    # model = get_model(df)
    results = []
    for h in hours:
        print(f'optimize for hour {h} ...')
        df_hour = df[df['hour'] == h]
        df_junction = group_by_junction(df_hour)
        print(df_junction)
        res = optimize_light_cycle({1: int(df_junction.loc[1]),2: int(df_junction.loc[2]),3: int(df_junction.loc[3])}, cycle)
        results.append({
            "hour": h,
            "vehicles_count": df_junction.to_dict(),
            "greens": res,
            "status": res['status'],
            # "pred": model.predict([[h,vehicles_count]])[0]
        })
        print(f'done for {h} ...')
        store_data(results, f'results{h}.csv')
    store_data(results, 'results.csv')
    end_time = time.time()
    print(f"Execution took {end_time - start_time:.2f} seconds.")
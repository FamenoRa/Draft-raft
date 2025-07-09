import time
import swifter
import pandas as pd
from store import store_data, retrieve_data
from QAOA_algo import optimize_light_cycle
from data_clean import get_data, group_by_junction_hour, group_by_junction
from data_training import train_congestion_model
from QAOA_algo import optimize_light_cycle

def get_model(df):
    return train_congestion_model(df)

if __name__ == "__main__":
    start_time = time.time()
    ratio = 0.6
    cycle = 60
    df = get_data()
    d = group_by_junction_hour(df)
    hours = range(24)
    model = get_model(df)
    results = []
    for h in hours:
        print(f'optimize for hour {h} ...')
        df_hour = df[df['hour'] == h]
        df_junction = group_by_junction(df_hour)
        main_vehicle_counts = int(df_junction[1])
        side_vehicle_counts = int(df_junction[2])
        vehicles_count = main_vehicle_counts+side_vehicle_counts
        res = optimize_light_cycle(vehicles_count, ratio, cycle)
        results.append({
            "hour": h,
            "main_green": res['main_green'],
            "side_green": res['side_green'],
            "vehicles_count": vehicles_count,
            "main_vehicle_counts": main_vehicle_counts,
            "side_vehicle_counts": side_vehicle_counts,
            "status": res['status'],
            # "pred": model.predict([[h,vehicles_count]])[0]
        })
        print(f'done for {h} ...')
    store_data(results, 'results.csv')
    end_time = time.time()
    print(f"Execution took {end_time - start_time:.2f} seconds.")
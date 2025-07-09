import json
import pandas as pd

def store_data(d, n):
    df = pd.DataFrame(d)
    df.to_csv(f'data/{n}', index=False)


def retrieve_data(n):
    return pd.read_csv(f'data/{n}')
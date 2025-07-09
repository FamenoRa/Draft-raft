from store import retrieve_data, retrieve_hour_vehicles


if __name__ == "__main__":
    res = retrieve_data('results.csv')
    r = retrieve_hour_vehicles(res, 7, 1)
    print(r.iloc[0]['main_green'])

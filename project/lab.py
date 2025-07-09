from store import store_data, retrieve_data

if __name__ == "__main__":
    res =  retrieve_data('results.csv')
    print(res.loc[0,'main_green'])

   
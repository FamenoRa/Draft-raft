from data_clean import get_data, group_by_junction_hour


if __name__ == "__main__":
    df = get_data()
    d = group_by_junction_hour(df)

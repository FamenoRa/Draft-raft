import json

def store_data(d):
    with open('data/output.json', 'w') as f:
        json.dump(d, f)

def retrieve_data():
    json.loads('output.json')
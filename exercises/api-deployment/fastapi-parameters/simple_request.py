import requests
import json

data = {'field_id': 71}

r = requests.post(
    'http://127.0.0.1:8000/add_data/simple_request/?query_param=3',
    data=json.dumps(data)
)

print(r.json())
## Named entity recognition and normalization from BERN pipeline
import requests
import json
import random
from tqdm import tqdm
import os

start = 81970 
os.environ['NO_PROXY'] = '127.0.0.1'

for source in ["pdf", "pmc"]:
    path = f"/home/akoksal/extra_space/COVID19-Bindings/Data/Abstract_Splitted/{source}_json.json"
    with open(path, "r") as f:
        data = json.load(f)
    if source == "pdf":
        keys = list(data.keys())[start:]
    else:
        keys = list(data.keys())
    for idx in tqdm(keys):
        results = []
        sentences = data[idx]
        for sentence in sentences:
            try:
                body_data = {"param": json.dumps({"text": sentence})}
                response = requests.post('http://127.0.0.1:8888', data=body_data)
                result_dict = response.json()
                results.append(result_dict)
            except Exception as e:
                print(f"Error in {idx}:",e)
                results.append({})
                continue
        with open(f"/home/akoksal/extra_space/COVID19-Bindings/Data/Abstract_Normalized/{source}/{idx}.json", "w") as f:
            json.dump(results, f)
    



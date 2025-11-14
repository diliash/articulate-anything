
import json
import os

with open("artiverse_data_full_filtered_valid_prev.json", 'r') as f:
    split_json = json.load(f)

category_models = {}

for key,value in split_json['test'].items():
    category_models[value["category"]] = category_models.get(value["category"], [])
    category_models[value["category"]].append(key)

print("Before:", len(split_json['test']))
split_json['test'] = {}



for category, models in category_models.items():
    print("before:", category, len(models))
    category_models[category] = models[:(len(models)+1)//2]
    print("after:", category, len(models))

for category, models in category_models.items():            
    for model in models:
        split_json['test'][model] = {"category": category}

with open("artiverse_data_full_filtered_valid.json", 'w') as f:
    json.dump(split_json, f, indent=4)
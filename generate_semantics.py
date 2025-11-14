import os
import json
from tqdm import tqdm

ARTIVERSE_PATH = "/cs/3dlg-jupiter-project/artiverse/singapo_data_full_filtered"

for category in reversed(os.listdir(ARTIVERSE_PATH)):
    category_path = os.path.join(ARTIVERSE_PATH, category)
    if not os.path.isdir(category_path):
        continue

    for model_id in tqdm(os.listdir(category_path)):
        model_path = os.path.join(category_path, model_id)
        if not os.path.isdir(model_path):
            continue

        desc_file = os.path.join(model_path, "train_v3.json")

        if not os.path.exists(desc_file):
            print(f"Description file not found for {model_id} in category {category}. Skipping.")
            continue

        with open(desc_file, "r") as f:
            descriptions = json.load(f)

        semantics_txt = ""
        
        for part in descriptions["diffuse_tree"]:
            part_name = part["name"]
            if part_name == "base":
                part_name = "body"
            semantics_txt += f'link_{part["id"]} {part["joint"]["type"]} {part_name}\n'

        semantics_file = os.path.join(model_path, "semantics.txt")
        with open(semantics_file, "w") as f:
            f.write(semantics_txt)
        

        
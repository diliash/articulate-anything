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

        links = []
        joints = []

        # Base link
        links.append('<link name="base"/>\n')


        # Load train_v3.json
        train_v3_file = os.path.join(model_path, "train_v3.json")

        if not os.path.exists(train_v3_file):
            continue
        with open(train_v3_file, "r") as f:
            train_v3_data = json.load(f)

        diffuse_tree = train_v3_data["diffuse_tree"]

        # Find root part (parent = -1)
        root_part = next(p for p in diffuse_tree if p["parent"] == -1)
        root_part_id = root_part["id"]

        # Map part_name -> link_name using semantics
        part_to_link = {}
        joint_id = 0

        semantic_txt = ""

        for i,part in enumerate(diffuse_tree):
            link_name = f"link_{part['id']}"

            part_to_link[part['id']] = link_name

            part_name = part['name']
            if part_name == "base":
                part_name = "body"

            

            visuals = ""
            for i in range(len(part['glbs'])):
                if part['glbs'][i].split('_')[-1][:-4] == "":
                    print(part['glbs'][i])
                # print(part['glbs'][i].split('_')[1][:-4])
                visuals += f"""
            <visual name="{part_name}-{i}">
                <origin xyz="0 0 0"/>
                <geometry>
                <mesh filename="textured_objs/original-{part['glbs'][i].split('part_')[-1][:-4]}.obj"/>
                </geometry>
            </visual>
            """

            # Build link definition
            links.append(f"""
        <link name="{link_name}">
            {visuals}   
        </link>
        """)

            semantic_txt+= f"{link_name} {part['joint']['type']} {part_name}\n"
            
        for part in diffuse_tree:

            child_name = part["name"]
            parent_index = part["parent"]

            # Skip root link: it attaches to base separately
            if parent_index == -1:
                continue

            parent_link = part_to_link[parent_index]

            child_link = part_to_link[part['id']]

            joints.append(f"""
        <joint type="fixed" name="joint_{joint_id}">
            <parent link="{parent_link}"/>
            <child link="{child_link}"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </joint>
        """)
            joint_id += 1

        # Add root joint (base → root link)
        root_link = part_to_link.get(root_part_id, "link_0")

        joints.insert(
            0,
            f"""
        <joint type="fixed" name="joint_root">
            <parent link="base"/>
            <child link="{root_link}"/>
            <origin xyz="0 0 0" rpy="1.5707963267948961 0 1.5707963267948961"/>
        </joint>
        """
        )

        # Final URDF assembly
        urdf_content = f"""<?xml version="1.0"?>
        <robot name="{model_id}">
        {''.join(links)}
        {''.join(joints)}
        </robot>
        """

        urdf_file = os.path.join(model_path, "mobility.urdf")
        with open(urdf_file, "w") as f:
            f.write(urdf_content)


        semantics_file = os.path.join(model_path, "semantics.txt")
        with open(semantics_file, "w") as f:
            f.write(semantic_txt)

        if os.path.exists(os.path.join(model_path, "semantics.json")):
            os.remove(os.path.join(model_path, "semantics.json"))

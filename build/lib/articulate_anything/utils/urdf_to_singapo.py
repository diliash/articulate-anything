import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from articulate_anything.api.odio_urdf import process_urdf
from articulate_anything.physics.pybullet_utils import get_bounding_box_center


def urdf_joint_to_global(joint, global_axis):
    """Convert a URDF joint definition to a global-space joint description."""
    quat = np.fromstring(joint["joint_origin"]["orientation"], sep=" ")
    R_joint_local = R.from_quat(quat).as_matrix()

    axis_local = np.fromstring(joint["joint_axis"], sep=" ")
    axis_local /= np.linalg.norm(axis_local)
    axis_global = (R_joint_local.T @ axis_local).round(7)

    lower = float(joint["joint_limit"]["lower"])
    upper = float(joint["joint_limit"]["upper"])
    if joint["joint_type"] == "revolute":
        lower, upper = np.degrees([lower, upper])

    return {
        "type": joint["joint_type"],
        "axis": {"origin": list(map(float, global_axis)), "direction": axis_global.tolist()},
        "range": [lower, upper],
    }


if __name__ == "__main__":
    robot = process_urdf(
        "partnet-mobility-v0-processed/partnet-mobility-v0/dataset/179/mobility.urdf"
    )

    # --- Compute global axes for all links ---
    topo_order = robot.compute_topological_order()
    link_states = robot.get_link_states(topo_order)

    global_axes = {"base": [0.0, 0.0, 0.0]}
    for link_name in topo_order:
        if link_name == "base":
            continue

        pos, quat = link_states[link_name][:2]
        parent = robot.get_parent_link_name(link_name)
        parent_pos, parent_quat = link_states[parent][:2]

        rot_M = R.from_quat(quat).as_matrix()
        axis = rot_M.T @ (np.array(pos) - np.array(parent_pos)) + np.array(global_axes[parent])
        global_axes[link_name] = axis.tolist()

    # --- Build joint parent-child info ---
    joint_info = {
        name: {
            "child": j.get_named_elements("Child")[0].link,
            "parent": j.get_named_elements("Parent")[0].link,
        }
        for name, j in robot.get_joints().items()
    }

    # --- Compute joint states ---
    joint_states = robot.get_joint_states()
    joints = {}
    for joint_name, state in joint_states.items():
        part_name = joint_info[joint_name]["child"]
        joints[part_name] = urdf_joint_to_global(state, global_axes[part_name])

    # --- Build link hierarchy ---
    children = {}
    name_to_id = {}
    for i, link_name in enumerate(l for l in robot.get_links() if l != "base"):
        parent = robot.get_parent_link_name(link_name)
        children.setdefault(parent, []).append(link_name)
        name_to_id[link_name] = i

    # --- Assemble final parts list ---
    parts = []
    for link_name in robot.get_links():
        if link_name == "base":
            continue

        parent = robot.get_parent_link_name(link_name)
        bbox_pos, bbox_size = robot.get_bounding_boxes(
            [link_name], include_dim=True
        )[link_name]

        bbox_center = get_bounding_box_center(bbox_pos[0], bbox_pos[1])

        bbox_size = [bbox_size["length"], bbox_size["width"], bbox_size["height"]]


        part_children = children.get(link_name, [])

        parts.append(
            {
                "id": name_to_id[link_name],
                "name": link_name,
                "parent": name_to_id.get(parent, -1),
                "children": [name_to_id[c] for c in part_children],
                "joint": joints.get(
                    link_name,
                    {
                        "type": "fixed",
                        "axis": {
                            "origin": [0.0, 0.0, 0.0],
                            "direction": [0.0, 0.0, 0.0],
                        },
                        "range": [0, 0.0],
                    },
                ),
                "aabb": {"center": list(bbox_center), "size": list(bbox_size)},
                "objs": [f"{link_name}_combined_mesh.obj"],
            }
        )

    # --- Save final structure ---
    json.dump({"diffuse_tree": parts}, open("tree.json", "w"), indent=4)
    print("✅ Saved to tree.json")

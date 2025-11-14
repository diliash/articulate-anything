import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from articulate import articulate
from omegaconf import OmegaConf
from dotenv import load_dotenv
from articulate_anything.utils.viz import (
    show_video, 
    display_code, 
    show_videos, 
    display_codes,
    show_images,
    get_frames_from_video,
)
from articulate_anything.utils.utils import load_config, join_path
from articulate_anything.utils.cotracker_utils import make_cotracker
from PIL import Image
import json



API_KEY = "YOUR-ACTUAL-API-KEY"
## we have our API key stored in a .env file
## Comment the `load_dotenv` and `os.environ.get` lines if you just want to use
## your API key directly
load_dotenv()
API_KEY = os.environ.get('API_KEY')


def process_model(category, model_id):
    task = model_id


    cfg = load_config("../../conf")


    modality = "image"
    prompt = f"/cs/3dlg-jupiter-project/artiverse/singapo_data_full_filtered/{category}/{model_id}/imgs/18.png"

    cfg = load_config()
    cfg.prompt = prompt
    cfg.modality = modality
    cfg.out_dir = join_path("results", "artiverse", task)



    use_cotracker = False # {True, False}
    mode = "image"
    actor_prompting_type = "basic" # {basic, incontext}
    critic_prompting_type = "incontext" # {basic, incontext}

    cfg.joint_actor.mode = mode
    cfg.joint_actor.use_cotracker = use_cotracker
    cfg.joint_actor.type = actor_prompting_type
    cfg.joint_actor.targetted_affordance = False

    cfg.joint_actor.examples_dir = "datasets/multi_modal_incontext_examples/joint_actor/in_context_actor_examples_datasets" ## Put your examples here


    cfg.joint_critic.mode = mode
    cfg.joint_critic.use_cotracker = use_cotracker
    cfg.joint_critic.type = critic_prompting_type

    cfg.joint_critic.examples_dir = "datasets/multi_modal_incontext_examples/joint_critic/in_context_examples_datasets" ## Put your examples here

    cfg.actor_critic.actor_only = True

    ## important to set correctly for the joint_critic to works properly
    ## this should have the same direction as the ground-truth video
    cfg.simulator.flip_video = False ## flip time for suitcase
    cfg.simulator.ray_tracing=False
    cfg.simulator.floor_texture = "plain"


    # cfg.category_selector.topk = 3
    cfg.category_selector.topk = 1 ## how many top categories should we search for an object template match
    # this is because PartNet-Mobility dataset categories labels are sparse and sometimes not great


    cfg.obj_selector.frame_index = 0

    cfg.actor_critic.max_iter = 2
    cfg.model_name = "gpt-4o-2024-11-20"


    cfg.api_key = API_KEY ## loaded from .env file

    steps = articulate(cfg)

with open("artiverse_data_full_filtered_valid.json", 'r') as f:
    split_json = json.load(f)

idx = 0

for category in os.listdir("/cs/3dlg-jupiter-project/artiverse/singapo_data_full_filtered/"):
    category_path = os.path.join("/cs/3dlg-jupiter-project/artiverse/singapo_data_full_filtered/", category)
    if not os.path.isdir(category_path):
        continue

    if category == "swivel_chair":
        continue

    i = 0
    for model_id in os.listdir(category_path):
        if model_id not in split_json['test']:
            continue
        model_path = os.path.join(category_path, model_id)
        if not os.path.isdir(model_path):
            continue
        

        print(f"Processing {category}/{model_id}")
        if not os.path.exists(os.path.join("results", "artiverse", model_id, "joint_actor")):
            process_model(category, model_id)
        idx+=1
        print(f"Processed {idx} models so far.")
        i+=1
        if i >=3:
            break
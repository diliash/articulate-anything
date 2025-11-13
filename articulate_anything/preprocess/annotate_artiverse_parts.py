import hydra
from omegaconf import OmegaConf, DictConfig
import os
from articulate_anything.utils.parallel_utils import process_tasks
from articulate_anything.agent.actor.mesh_retrieval.artiverse_mesh_annotator import ArtiverseMeshAnnotator


def annotate_artiverse_part(obj_id, gpu_id, cfg):
    artiverse_dir = join_path(cfg.dataset_dir, obj_id)
    cfg = OmegaConf.create(cfg)  # copy the configuration
    cfg.out_dir = artiverse_dir
    artiverse_annotator = ArtiverseMeshAnnotator(cfg)
    artiverse_annotator.generate_prediction(artiverse_dir=artiverse_dir)


@ hydra.main(version_base=None, config_path="../../conf", config_name="config")
def preprocess_artiverse(cfg: DictConfig):
    obj_ids = []
    for category in os.listdir(cfg.dataset_dir):
        category_path = os.path.join(cfg.dataset_dir, category)
        if not os.path.isdir(category_path):
            continue

        for model_id in os.listdir(category_path):
            model_path = os.path.join(category_path, model_id)
            if not os.path.isdir(model_path):
                continue
            obj_ids.append(os.path.join(category, model_id))

    process_tasks(obj_ids, annotate_artiverse_part, num_workers=cfg.parallel,
                  max_load_per_gpu=cfg.max_load_per_gpu,
                  cfg=cfg)


if __name__ == "__main__":
    preprocess_artiverse()

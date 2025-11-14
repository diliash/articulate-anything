import argparse
import trimesh
import xml.etree.ElementTree as ET
import copy
import base64
import cv2
import numpy as np
import sapien.core as sapien
import re
import logging
import os
import glob
from PIL import Image
from articulate_anything.agent.agent import Agent
import json
from articulate_anything.utils.utils import (
    save_json,
    join_path,
    file_to_string,
)

ARTIVERSE_MESH_ANNOTATOR_SYSTEM_INSTRUCTION = """
You are a helpful part annotator whose job is to give each link specified in the semantic label more detailed descriptions. You are given a few views of the assembled object.
Details include: (material properties, shape, function, etc.)

The semantic label file look like this:

```
link_0 <joint_type> <semantic_label>
link_1 <joint_type> <semantic_label>
...
```

return a detailed description of each part mentioned in the semantic label file in JSON format:
```json
{
    "reasoning": "I see... ",
    "annotation": {
        "object": "<overall object description>", # eg. a wooden cabinet with a drawer and a door.
        "link_0": "<description>", # eg. "A wooden drawer that slides in and out."
        "link_1": "<description>",
    }
    ...
}
```
Tips:

- Do not label parts as things they are not. If the semantic says it is a drawer, do not label it as a door.
- If you cannot see the part in the frames, simply say: "a <semantic_label> for <the object that you see>"
- Give EXACTLY the number of parts mentioned in the semantic label file. No more, no less.
- Do not speculate on the motion of the parts if you do not see any motion in the frames.
- Use strong, confident language.
- You might comment on the material, shap, texture but do NOT comment on the color as we will not render the color in the final output.
- Later, we will run a CLIP model to get embeddings for each part mesh, and retrieve the part meshes as part of our text-to-3D pipeline.
    - Thus, you need to be specific in your `link` descriptions to ensure the CLIP model can accurately retrieve the correct part mesh.
    For example, there are `toilet_body` that are intended to be a public_toilet or be wall-mounted. These toilets would visually look very different
    from say a in-home toilet. Your descriptions should be specific enough to differentiate between these two types of toilets.
"""


class ArtiverseMeshAnnotator(Agent):
    OUT_RESULT_PATH = "object_annotation_gemini.json"

    def _make_system_instruction(self):
        return ARTIVERSE_MESH_ANNOTATOR_SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, artiverse_dir):
        object_image_paths = [f"{artiverse_dir}/imgs/18.png", f"{artiverse_dir}/imgs/19.png",]
        object_image_prompt = ['Images of the Assembled Object:\n']
        for object_image_path in object_image_paths:
            object_image_prompt.append(Image.open(object_image_path))


        semantics = file_to_string(join_path(artiverse_dir, 'semantics.txt'))
        message = """
            Please help me with this object:

            ```
            <semantics>
            ```
            """
        message = message.replace("<semantics>", "".join(semantics))
        prompt = [message] + object_image_prompt
        return prompt

    def parse_response(self, response, **kwargs):
        json_str = response.text.strip().strip('```json').strip()

        parsed_response = json.loads(json_str, strict=False)
        logging.info(f"PartNet mesh annotator response: {parsed_response}")

        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import cv2
import random
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog
from .load_vg_json import load_vg_json

SPLITS = {
    "visual_genome_train": ("vg/images", "vg/annotations/train.json"),
    "visual_genome_val": ("vg/images", "vg/annotations/val.json"),
}

for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    json_file = os.path.join("datasets", json_file)
    image_root = os.path.join("datasets", image_root)
    DatasetCatalog.register(
        key,
        lambda key=key, json_file=json_file, image_root=image_root: load_vg_json(
            json_file, image_root, key
        ),
    )
    MetadataCatalog.get(key).set(
        json_file=json_file, image_root=image_root
    )
# #Visualizing the Train Dataset
# dataset_dicts = load_vg_json("../datasets/vg/annotations/train.json","../datasets/vg/images","visual_genome_train")
# #Randomly choosing 3 images from the Set
# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     print(img)
#     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("visual_genome_train"))
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imwrite("res.png",vis.get_image()[:, :, ::-1])
# meta=MetadataCatalog.get("visual_genome_train")
# thing_classes=set()
# if hasattr(meta, "thing_classes"):
#     thing_classes.update(meta.thing_classes)
# print(meta)

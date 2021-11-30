import numpy as np
from configs.path_config import SHAPENETCLASSES
from configs.path_config import ScanNet_OBJ_CLASS_IDS as OBJ_CLASS_IDS, ShapeNetIDMap
import torch


class ScannetConfig(object):
    def __init__(self):
        self.num_class = len(OBJ_CLASS_IDS)
        self.num_heading_bin = 12
        self.num_size_cluster = len(OBJ_CLASS_IDS)

        self.type2class = {
            SHAPENETCLASSES[cls]: index for index, cls in enumerate(OBJ_CLASS_IDS)}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.class_ids = OBJ_CLASS_IDS
        self.shapenetid2class = {class_id: i for i,
                                 class_id in enumerate(list(self.class_ids))}
        self.shapenet_id_map = ShapeNetIDMap
        

import numpy as np
import torch
import pytorch3d
from torch.utils.data.dataset import Dataset
import trimesh
from mesh_to_sdf import mesh_to_voxels

import transforms


class ShapeNetCoreDataset(Dataset):
    def __init__(self,
                 config,
                 *args,
                 **kwargs):

        super(ShapeNetCoreDataset, self).__init__()
        self.root = config['data']['shapenet_path']
        self.num_sample_points = config['data']['num_points']
        self.random_rotation = config['data'].get(
            'apply_random_rotation', False)
        self.random_cropping = config['data'].get(
            'apply_random_cropping', False)
        self.shape_net_core = pytorch3d.datasets.ShapeNetCore(root=self.root)

    def __len__(self):
        return len(self.shape_net_core)

    def __getitem__(self, index):
        # get model
        model = self.shape_net_core[index]
        label = model['label']

        mesh = trimesh.base.Trimesh(
            verices=model['verts'], faces=model['faces'])

        if self.random_rotation:
            mesh = transforms.random_rotation(mesh)

        if self.random_cropping:
            mesh = transforms.random_crop(mesh)

        pc = mesh.sample(self.num_sample_points)

        occ_grid_size = None  # TODO: dim of occupancy grid
        occ_grid = mesh_to_voxels(mesh, occ_grid_size, pad=True)

        return {
            'points': pc,
            'label': label,
            'occupancy_grid': occ_grid
        }

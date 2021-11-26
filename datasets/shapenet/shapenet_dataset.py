import os
from pathlib import Path

import numpy as np
import torch
import pytorch3d
from torch.utils.data.dataset import Dataset

from external import binvox_rw


def iterate_shapenet(shapenet_path):
    shape_dirs = []
    for catid_path in Path(shapenet_path).iterdir():
        if catid_path.is_dir() and not catid_path.name.startswith('.'):
            for shapeid_path in catid_path.iterdir():
                if shapeid_path.is_dir() and not shapeid_path.name.startswith('.'):
                    shape_dirs.append((catid_path.name, shapeid_path.name))

    return shape_dirs


class ShapeNetCoreDataset(Dataset):
    def __init__(self,
                 config,
                 *args,
                 **kwargs):

        super(ShapeNetCoreDataset, self).__init__()
        self.root = config['data']['shapenet_path']
        self.point_root = os.path.join(self.root, 'point')
        self.pointcloud_root = os.path.join(self.root, 'pointcloud')
        self.voxel_root = os.path.join(self.root, 'voxel')
        self.waterthight_simplified_root = os.path.join(
            self.root, 'watertight_scaled_simplified')

        self.shapes_index = iterate_shapenet(self.point_root)

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

        points_path = os.path.join(
            self.point_root, *self.shapes_index[index], '.npz')
        points_dict = np.load(points_path)
        points = points_dict['points']

        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        occupancies = occupancies.astype(np.float32)

        pointcloud_path = os.path.join(
            self.pointcloud_root, *self.shapes_index[index], '.npz')
        pointcloud = np.load(pointcloud_path)

        voxel_path = os.path.join(
            self.voxel_root, *self.shapes_index[index], '.binvox')

        with open(voxel_path, mode='r') as f:
            voxels = binvox_rw.read_as_3d_array(f)

        """
        mesh = trimesh.base.Trimesh(
            verices=model['verts'], faces=model['faces'])

        if self.random_rotation:
            mesh = transforms.random_rotation(mesh)

        if self.random_cropping:
            mesh = transforms.random_crop(mesh)

        pc = mesh.sample(self.num_sample_points)

        occ_grid_size = 128  # TODO: is this in somewhere in the config
        occ_grid = mesh_to_voxels(mesh, occ_grid_size, pad=True)

        pc = pc.astype(np.float32)
        occ_grid = occ_grid.astype(np.float32)
        """

        return {
            'object_points': points,
            'objcet_occupancies': occupancies,
            'object_pointcloud': pointcloud,
            'object_voxels': voxels,
            'label': label,
        }

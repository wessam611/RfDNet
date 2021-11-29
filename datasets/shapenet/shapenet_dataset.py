import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import trimesh

from . import binvox_rw, pc_util, transforms


class ShapeNetCoreDataset(Dataset):
    def __init__(self,
                 config,
                 dataset_config,
                 *args,
                 **kwargs):

        super(ShapeNetCoreDataset, self).__init__()
        self.config = config
        self.dataset_config = dataset_config
        self.root = Path(config['data']['shapenet_path'])
        self.shape_index = self.get_shapenet_index()
        self.num_sample_points = config['data']['num_points']
        self.points_unpackbits = config['data']['points_unpackbits']
        self.random_rotation = config['data'].get(
            'apply_random_rotation', False)
        self.random_cropping = config['data'].get(
            'apply_random_cropping', False)
        self.random_cropping_mode = config['data'].get(
            'random_cropping_mode', None)

    def __len__(self):
        return len(self.shape_index)

    def __getitem__(self, index):
        shape_dict = self.shape_index[index]

        label = shape_dict['cat_id']
        label = str(int(label))
        label = self.dataset_config.shapenet_id_map[label]
        label = self.dataset_config.type2class[label]

        # read points and occupancies
        points_dict = np.load(os.path.join(self.root, shape_dict['point']))
        points = points_dict['points'].astype(np.float32)
        occupancies = points_dict['occupancies']
        if self.points_unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # read pointcloud
        pointcloud = np.load(os.path.join(self.root, shape_dict['pointcloud']))
        pointcloud = pointcloud['points'].astype(np.float32)

        # apply augmentation according to config
        if self.random_rotation:
            pointcloud = transforms.random_rotation(pointcloud)

        if self.random_cropping:
            pointcloud = transforms.random_crop(
                pointcloud, self.random_cropping_mode)

        # sample N points from pointcloud
        pointcloud = pc_util.random_sampling(
            pointcloud, self.num_sample_points)

        # read voxels
        voxel_file = os.path.join(self.root, shape_dict['voxel'])
        with open(voxel_file, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f).data

        # covert to PyTorch tensor
        points = torch.from_numpy(points)
        occupancies = torch.from_numpy(occupancies)
        pointcloud = torch.from_numpy(pointcloud)
        voxels = torch.from_numpy(voxels)

        return {
            'object_points': points,
            'object_occupancies': occupancies,
            'object_pointcloud': pointcloud,
            'object_voxels': voxels,
            'label': label,
        }

        """
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

    def get_shapenet_index(self) -> List[dict]:
        shapenet_path = self.root / 'point'
        shape_index = []

        cat_ids = [p for p in shapenet_path.iterdir()
                   if p.is_dir and not p.name.startswith('.')]

        for cat_dir in cat_ids:
            shape_ids = [p for p in cat_dir.iterdir()
                         if p.is_file and not p.name.startswith('.')]

            for shape_id in shape_ids:
                shape_index.append({
                    'cat_id': cat_dir.name,
                    'shape_id': shape_id.name,
                    'point': os.path.join('point', cat_dir.name, shape_id.name),
                    'pointcloud': os.path.join('pointcloud', cat_dir.name, shape_id.name),
                    'voxel': os.path.join('voxel', '16', cat_dir.name, f'{shape_id.stem}.binvox'),
                    'watertight_scaled_simplified': os.path.join('watertight_scaled_simplified',
                                                                 cat_dir.name, shape_id.name),
                })

        return shape_index


# NOTE: make sure this is commented out when using the dataset
"""
if __name__ == '__main__':
    # A simple test

    config = {
        'data': {
            'shapenet_path': 'datasets/ShapeNetv2_data',
            'num_points': 10_000,
            'points_unpackbits': True,
            'apply_random_cropping': True,
            'random_cropping_mode': 4
        }
    }

    dataset = ShapeNetCoreDataset(config, dataset_config)

    shape = dataset[1500]
    print('Label:', shape['label'])
    print('Points:', shape['object_points'].shape,
          shape['object_points'].dtype)

    print('Occupancies:', shape['object_occupancies'].shape,
          shape['object_occupancies'].dtype)

    print('Pointcloud', shape['object_pointcloud'].shape,
          shape['object_pointcloud'].dtype)

    print('Voxels', shape['object_voxels'].shape,
          shape['object_voxels'].dtype)

    pcl = trimesh.points.PointCloud(shape['object_pointcloud'].numpy())
    pcl.show()
"""

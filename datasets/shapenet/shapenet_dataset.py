import os
from pathlib import Path
from typing import List
from random import randint

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import trimesh
from net_utils.transforms import SubsamplePoints

from . import binvox_rw, pc_util, transforms

from utils.read_and_write import read_json


class ShapeNetCoreDataset(Dataset):
    def __init__(self,
                 cfg,
                 mode):

        super(ShapeNetCoreDataset, self).__init__()
        self.dataset_config = cfg.dataset_config
        self.root = Path(cfg.config['data']['shapenet_path'])
        self.shape_index = self.get_shapenet_index(
            cfg.config['data']['split'], mode)
        self.num_sample_points = cfg.config['data']['num_point']
        self.points_unpackbits = cfg.config['data']['points_unpackbits']
        self.points_sampler = SubsamplePoints(
            cfg.config['data']['points_subsample'], mode)
        self.random_rotation = cfg.config['data'].get(
            'apply_random_rotation', False)
        self.random_cropping = cfg.config['data'].get(
            'apply_random_cropping', False)
        self.random_cropping_mode = cfg.config['data'].get(
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
        sampled = self.points_sampler({
            'points': points,
            'occ': occupancies
        })

        points = sampled['points']
        occupancies = sampled['occ']

        # read pointcloud
        pcl_path = Path(shape_dict['pointcloud'])
        shape_id = pcl_path.name
        cat_id = pcl_path.parent.name
        pcl_path = pcl_path.parent.parent.parent / \
            'partial_pointcloud' / cat_id / shape_id

        pointcloud = np.load(os.path.join(self.root, pcl_path))
        rand_idx = randint(0, 3)
        pointcloud = pointcloud[f'arr_{rand_idx}']

        # apply augmentation according to cfg.config
        if self.random_rotation:

            data = {
                'pointcloud': pointcloud,
                'points': points,
            }

            pointcloud, points = transforms.random_rotation(
                data, max_rotation_angle=0.05)

        if self.random_cropping:
            pointcloud = transforms.random_crop(
                pointcloud, self.random_cropping_mode)

        # sample N points from pointcloud
        pointcloud = pc_util.random_sampling(
            pointcloud, self.num_sample_points)

        pointcloud = pointcloud.astype(np.float32)
        points = points.astype(np.float32)

        # read voxels
        voxel_file = os.path.join(self.root, shape_dict['voxel'])
        with open(voxel_file, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f).data

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

    def get_shapenet_index(self, split_path, mode) -> List[dict]:
        return read_json(f'{split_path}shapenet_{mode}.json')


# NOTE: make sure this is commented out when using the dataset
"""
if __name__ == '__main__':
    # A simple test

    cfg.config = {
        'data': {
            'shapenet_path': 'datasets/ShapeNetv2_data',
            'num_points': 10_000,
            'points_unpackbits': True,
            'apply_random_cropping': True,
            'random_cropping_mode': 4
        }
    }

    dataset = ShapeNetCoreDataset(cfg.config)

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

import os
from pathlib import Path

import numpy as np

from utils.read_and_write import read_json, write_json


def split_data():
    shapenet_train_split = 'datasets/splits/shapenet/shapenet_train.json'
    shapenet_val_split = 'datasets/splits/shapenet/shapenet_val.json'
    shapenet_overfit_split = 'datasets/splits/shapenet/shapenet_overfit.json'

    if not os.path.exists(os.path.dirname(shapenet_train_split)):
        os.makedirs(os.path.dirname(shapenet_train_split))
    if not os.path.exists(os.path.dirname(shapenet_val_split)):
        os.makedirs(os.path.dirname(shapenet_val_split))
    if not os.path.exists(os.path.dirname(shapenet_overfit_split)):
        os.makedirs(os.path.dirname(shapenet_overfit_split))
    
    shapenet_path =  Path('datasets/ShapeNetv2_data/point')
    shape_index = []

    cat_ids = [p for p in shapenet_path.iterdir()
                if p.is_dir and not p.name.startswith('.')]

    for cat_dir in cat_ids:
        shape_ids = [p for p in cat_dir.iterdir()
                        if p.is_file and not p.name.startswith('.')]

        for shape_id in shape_ids:
            shape_index.append({
                'cat_id': cat_dir.name,
                'shape_id': shape_id.name.split('.')[0],
                'point': os.path.join('point', cat_dir.name, shape_id.name),
                'pointcloud': os.path.join('pointcloud', cat_dir.name, shape_id.name),
                'voxel': os.path.join('voxel', '16', cat_dir.name, f'{shape_id.stem}.binvox'),
                'watertight_scaled_simplified': os.path.join('watertight_scaled_simplified',
                                                                cat_dir.name, shape_id.name),
            })
    indices = np.arange(len(shape_index))
    np.random.shuffle(indices)
    split_train = int(len(shape_index)*0.8)
    split_overfit = 20
    split_val = len(shape_index) - split_train - split_overfit
    shape_index_splits = np.split(indices, [split_train, split_train+split_val, len(shape_index)])
    train = [shape_index[i] for i in shape_index_splits[0]]
    val = [shape_index[i] for i in shape_index_splits[1]]
    overfit = [shape_index[i] for i in shape_index_splits[2]]
    write_json(shapenet_train_split, train)
    write_json(shapenet_val_split, val)
    write_json(shapenet_overfit_split, overfit)
    
    



if __name__ == '__main__':
    split_data()
import os
from pathlib import Path
import argparse

import numpy as np
import torch
from torch._C import dtype
from tqdm import tqdm

from configs.config_utils import CONFIG


from models.optimizers import load_optimizer, load_scheduler, load_bnm_scheduler
from net_utils.utils import load_device, load_model, load_trainer, load_dataloader
from net_utils.utils import CheckpointIO
from train_epoch import train
from configs.config_utils import mount_external_config


def parse_args():
    parser = argparse.ArgumentParser('ShapeNet encodings')
    parser.add_argument('--config', type=str, default='configs/config_files/ISCNet_encodings.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--path', type=str, default='datasets/ShapeNetv2_data/encodings')
    return parser.parse_args()


@torch.no_grad()
def generate_encodings(cfg, path):

    '''Mount external config data'''
    cfg = mount_external_config(cfg)


    shapenet_path =  Path('datasets/ShapeNetv2_data/pointcloud')
    encodings = {}

    device = load_device(cfg)
    net = load_model(cfg, device=device)

    cat_ids = [p for p in shapenet_path.iterdir()
                if p.is_dir and not p.name.startswith('.')]

    if not os.path.exists(path):
        os.makedirs(path)
    for cat_dir in cat_ids:
        cat_save_path = os.path.join(path, cat_dir.name)
        if not os.path.exists(cat_save_path):
            os.makedirs(cat_save_path)
        shape_ids = [p for p in cat_dir.iterdir()
                        if p.is_file and not p.name.startswith('.')]
        print(f'processing class {cat_dir}')
        for shape_id in tqdm(shape_ids):
            pc_path = os.path.join('datasets/ShapeNetv2_data/pointcloud', cat_dir.name, shape_id.name)
            save_path = os.path.join(path, cat_dir.name, shape_id.name)
            # this line should be changed in case we decided to use preprocessed points
            # (simulating scanned PC)
            input_points = np.load(pc_path)['points']
            indx = np.random.choice(input_points.shape[0], 
                                       size=cfg.config['data']['num_point']*
                                             cfg.config['data']['num_samples'], 
                                       replace=False)
            input_points = input_points[indx].reshape(
                cfg.config['data']['num_samples'], cfg.config['data']['num_point'], 3
            )
            input_points = torch.from_numpy(input_points).type(torch.float32)
            input_points = input_points.to(device)
            _, curr_enc = net.module.class_encode(input_points)
            curr_enc = torch.mean(curr_enc, dim=0)
            curr_enc.detach()
            encodings[save_path] = curr_enc.cpu().numpy()
    for curr_path, encoding in encodings.items():
        np.savez(curr_path, encoding=encoding)


if __name__ == '__main__':
    args = parse_args()
    cfg = CONFIG(args.config)
    path = args.path
    generate_encodings(cfg, path)

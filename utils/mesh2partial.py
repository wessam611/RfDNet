"""
Usage: python mesh2partial.py PATH_TO_MESH_FOLDER (i.e., datasets/ShapeNetv2_data/watertight_scaled_simplified/)
"""

from pathlib import Path
import argparse

import numpy as np
import trimesh
from tqdm import tqdm


def sample_pointcloud(mesh_file):
    mesh = trimesh.load_mesh(mesh_file)

    samples, face_indices = mesh.sample(50_000, return_index=True)
    normals = mesh.face_normals[face_indices]

    x_rand = list(np.random.rand(4))
    z_rand = list(np.random.rand(4))
    viewpoints = [np.array([x, 0., z]) for x, z in zip(x_rand, z_rand)]

    ret_list = []
    for viewpoint in viewpoints:
        viewpoint = viewpoint / np.linalg.norm(viewpoint)

        dot_product = np.dot(viewpoint, normals.T)
        angles = np.arccos(dot_product)

        drop_idx = np.where(angles > (np.pi / 3.))
        points = np.delete(samples, drop_idx, axis=0)

        ret_list.append(points)

    return ret_list


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str)
    args = ap.parse_args()

    mesh_root = Path(args.path)

    out_path = mesh_root.parent.joinpath('partial_pointclouds')
    if not out_path.exists():
        out_path.mkdir()

    cat_dirs = [p for p in mesh_root.iterdir() if p.is_dir()
                and not p.name.startswith('.')]
    for cat_dir in cat_dirs:

        print(f'PROCESSING: {cat_dir.name}')

        save_path = out_path / cat_dir.name
        if not save_path.exists():
            save_path.mkdir()

        shape_files = [p for p in cat_dir.iterdir() if p.is_file()
                       and p.suffix == '.off']
        for file in tqdm(shape_files):
            pointclouds = sample_pointcloud(file)

            filename = save_path / f'{file.stem}.npz'
            np.savez(filename, *pointclouds)


if __name__ == '__main__':
    main()

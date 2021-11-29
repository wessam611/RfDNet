import numpy as np
from trimesh.base import Trimesh
from scipy.spatial.transform import Rotation


def random_rotation(mesh: Trimesh):
    R = Rotation.random(1).as_matrix().reshape(3, 3)
    T = np.zeros((4, 4), dtype=np.float32)
    T[:3, :3] = R
    T[3, 3] = 1
    mesh.apply_transform(T)
    return mesh


def random_crop(mesh: Trimesh):
    raise NotImplementedError()

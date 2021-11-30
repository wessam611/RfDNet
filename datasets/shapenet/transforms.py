import random

import numpy as np
from pc_util import (
    rotate_point_cloud,
    point_cloud_to_bbox,
)


def random_rotation(pointcloud: np.array) -> np.array:
    """
    Randomly rotate the given pointcloud.

    :param pointcloud: The pointcloud to be rotated
    :return pointcloud: The rotated pointcloud
    """

    pointcloud, _ = rotate_point_cloud(pointcloud)
    return pointcloud


def random_crop(pointcloud: np.array, fraction: int) -> np.array:
    """
    Randomly crop the given pointcloud by a given fraction.

    :param pointcloud: The pointcloud to be cropped
    :param fraction: either 2 to crop one random half or 4 to crop one random quater.

    :return pointcloud: The cropped pointcloud
    """

    pcl_bbox = point_cloud_to_bbox(pointcloud)
    center = pcl_bbox[:3]

    if fraction == 2:
        axis = random.choice([0, 1, 2])
        block = random.choice(['less', 'greater'])

        if block == 'less':
            print(center)
            pointcloud = pointcloud[pointcloud[:, axis] < center[axis]]
        else:
            pointcloud = pointcloud[pointcloud[:, axis] > center[axis]]

    elif fraction == 4:
        axis_1 = random.choice([0, 1, 2])
        axis_2 = random.choice(list({0, 1, 2} - {axis_1}))
        print(axis_1, axis_2)
        block_1 = random.choice(['less', 'greater'])
        block_2 = random.choice(['less', 'greater'])

        if block_1 == 'less':
            if block_2 == 'less':
                pointcloud = pointcloud[(pointcloud[:, axis_1] < center[axis_1]) | (
                    pointcloud[:, axis_2] < center[axis_2])]
            else:
                pointcloud = pointcloud[(pointcloud[:, axis_1] < center[axis_1]) | (
                    pointcloud[:, axis_2] > center[axis_2])]
        else:
            if block_2 == 'less':
                pointcloud = pointcloud[(pointcloud[:, axis_1] > center[axis_1]) | (
                    pointcloud[:, axis_2] < center[axis_2])]
            else:
                pointcloud = pointcloud[(pointcloud[:, axis_1] > center[axis_1]) | (
                    pointcloud[:, axis_2] > center[axis_2])]

    return pointcloud

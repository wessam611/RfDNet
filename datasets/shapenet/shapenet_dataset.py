
import torch
import pytorch3d
from torch.utils.data.dataset import Dataset


class ShapeNetCoreDataset(Dataset):
    def __init__(self, root):
        super(ShapeNetCoreDataset, self).__init__()
        self.root = root
        self.shape_net_core = pytorch3d.datasets.ShapeNetCore(root=root)

    def __len__(self):
        return len(self.shape_net_core)

    def __getitem__(self, index):
        shape = self.shape_net_core[index]

        # TODO: transform to pointcloud, perform data augmentation:
        # - random cropping
        # - drop points (make the pc sparser)

        return shape

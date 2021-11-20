
import torch
import pytorch3d
from torch.utils.data.dataset import Dataset


class ShapeNetCoreDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ShapeNetCoreDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.shape_net_core = pytorch3d.datasets.ShapeNetCore(root=root)

    def __len__(self):
        return len(self.shape_net_core)

    def __getitem__(self, index):
        shape = self.shape_net_core[index]

        if self.transform is not None:
            shape = self.transform(shape)

        return shape

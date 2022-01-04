from models.registers import MODULES
import torch
from torch import nn
from models.iscnet.modules.layers import ResnetPointnet


@MODULES.register_module
class ClassEncoder(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(ClassEncoder, self).__init__()
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=3,#self.input_feature_dim + 3 + 128, # should be gotten from dataset config
                                      hidden_dim=cfg.config['data']['hidden_dim'])
        self.linear = nn.Linear(cfg.config['data']['c_dim'], cfg.dataset_config.num_class)

    def forward(self, pc):
        features = self.encoder(pc)
        logits = self.linear(features)
        return logits, features
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
        self.feat_loss = cfg.config['model']['class_encode'].get('feature_wise_loss', False)
        self.feat_samples = cfg.config['model']['class_encode'].get('feature_wise_samples', 0)
        self.c_dim = cfg.config['data']['c_dim']

    def forward(self, pc, point_seg_mask=None):
        B, T, F = pc.shape
        features_wise_loss = None
        if (self.feat_loss):
            features_samples = torch.zeros((self.feat_samples+1, B, self.c_dim), device=pc.device)
            for i in range(self.feat_samples):
                mask = torch.randn(pc.shape[:2], device=pc.device) > 0.8
                if point_seg_mask is None:
                    features_samples[i] = self.encoder(pc, point_seg_mask=mask)
                else:
                    features_samples[i] = self.encoder(pc, point_seg_mask=point_seg_mask*mask)
            features_samples[-1] = self.encoder(pc, point_seg_mask=point_seg_mask)
            inner_object_var = features_samples.var(dim=0).sum(-1).mean()
            cross_object_var = features_samples.var(dim=1).sum(-1).mean()
            th = torch.Tensor([0.]).to(pc.device)[0]
            features_wise_loss = torch.maximum(inner_object_var - cross_object_var + 1, th)
            features = features_samples[-1]
        else:
            features = self.encoder(pc, point_seg_mask=point_seg_mask)
        logits = self.linear(features)
        return logits, features, features_wise_loss
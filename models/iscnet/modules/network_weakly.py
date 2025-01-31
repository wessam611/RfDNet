# ISCNet: model loader
# author: ynie
# date: Feb, 2020

from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from torch import nn
import torch.nn.functional as F
from net_utils.nn_distance import nn_distance
import numpy as np
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
from external.common import compute_iou
from net_utils.libs import flip_axis_to_depth, extract_pc_in_box3d, flip_axis_to_camera
from torch import optim
from models.loss import chamfer_func
from net_utils.box_util import get_3d_box
from external.group_loss import GTG, get_labeled_and_unlabeled_points

from .network import ISCNet


@METHODS.register_module
class ISCNet_WEAK(BaseNetwork):
    def __init__(self, cfg):
        """
        load submodules for the network.
        :param config: customized configurations.
        """
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        phase_names = []
        if cfg.config[cfg.config['mode']]['phase'] in ['prior']:
            phase_names += ['completion', 'class_encode']

        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')
        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1)))
            '''load corresponding loss functions'''
        for metric_name in cfg.config['val'].get('metrics', []):
            metric_fn = cfg.config['val']['metrics'][metric_name]
            setattr(self, metric_name + '_metric', LOSSES.get(metric_fn, 'Null')())

        # Init group loss
        self.gtg = GTG(self.cfg.config['data']['num_classes'])

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    # def sample_for_prior(self, inputs, sample_size):
    #     # temporary naive implementation should be either random
    #     # or sampled by removing some chungs of the input pC so it has holes
    #     # instead of being evenly distributed
    #     return inputs[:, 0:sample_size]
    def forward(self, data, export_shape=False):
        """
        Forward pass of the network
        :param data (dict): contains the data for training.
        :param export_shape: if output shape voxels for visualization
        :return: end_points: dict
        """
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'prior':
            pc = data['object_pointcloud']
            input_points = pc  # self.sample_for_prior(pc, 256) # not sure where the no. of points per object is set
            logits, features_for_completion, feature_wise_loss = self.class_encode(input_points)
            completion_loss, shape_example = self.completion.compute_loss(features_for_completion,
                                                                          data['object_points'],  # just a 3D grid
                                                                          data['object_occupancies'],  # Labels in out
                                                                          data['label'],  # class labels for ShapeNet
                                                                          export_shape)
            # NOTE: end_points, BATCH_PROPOSAL_IDs removed
            return logits, features_for_completion, completion_loss, shape_example, feature_wise_loss
        else:
            pass
            # not yet decided
            # return super(ISCNet_WEAK, self).forward(data, export_shape)

    def loss(self, est_data, gt_data):
        """
        calculate loss of est_out given gt_out.
        """
        # if self.cfg.config[self.cfg.config['mode']]['phase'] == 'prior':
        completion_loss = self.completion_loss(est_data[2])

        # Group loss
        probs, features, _, _, feats_loss = est_data
        labs, L, U = get_labeled_and_unlabeled_points(gt_data['label'],
                                                      self.cfg.config['train']['num_labeled_points_per_class'],
                                                      self.cfg.config['data']['num_classes'])

        probs_gtg = F.softmax(probs, dim=1)
        probs_gtg, W = self.gtg(features, features.shape[0], labs, L, U, probs_gtg)
        probs_gtg = torch.log(probs_gtg + 1e-12)

        class_loss = self.class_encode_loss((probs, probs_gtg, feats_loss), gt_data)
        total_loss = {'completion_loss': completion_loss.item(),
                      'class_loss': class_loss.item(),
                      'total': class_loss + completion_loss}
        return total_loss

    def compute_metrics(self, est_data, gt_data):
        """
        """
        metrics = {'cls_acc_met': self.cls_acc_metric(est_data, gt_data)}
        return metrics

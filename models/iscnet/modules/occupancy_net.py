# Occupancy Networks
import torch
from torch._C import dtype
import torch.nn as nn
from models.registers import MODULES
import torch.distributions as dist
from models.iscnet.modules.encoder_latent import Encoder_Latent
from models.iscnet.modules.occ_decoder import DecoderCBatchNorm
from torch.nn import functional as F
from external.common import make_3d_grid
import numpy as np


@MODULES.register_module
class ONet(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, cfg, optim_spec=None):
        super(ONet, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Parameter Configs'''
        decoder_kwargs = {}
        encoder_latent_kwargs = {}
        self.z_dim = cfg.config['data']['z_dim']
        dim = 3
        self.use_cls_for_completion = cfg.config['data']['use_cls_for_completion']
        if not cfg.config['data']['skip_propagate']:
            c_dim = self.use_cls_for_completion*cfg.dataset_config.num_class + 128
        else:
            c_dim = self.use_cls_for_completion * \
                cfg.dataset_config.num_class + cfg.config['data']['c_dim']
        self.threshold = cfg.config['data']['threshold']

        '''Module Configs'''
        if self.z_dim != 0:
            self.encoder_latent = Encoder_Latent(
                dim=dim, z_dim=self.z_dim, c_dim=c_dim, **encoder_latent_kwargs)
        else:
            self.encoder_latent = None

        self.decoder = DecoderCBatchNorm(
            dim=dim, z_dim=self.z_dim, c_dim=c_dim, **decoder_kwargs)

        '''Mount mesh generator'''
        if 'generation' in cfg.config and cfg.config['generation']['generate_mesh']:
            from models.iscnet.modules.generator import Generator3D
            self.generator = Generator3D(self,
                                         threshold=cfg.config['data']['threshold'],
                                         resolution0=cfg.config['generation']['resolution_0'],
                                         upsampling_steps=cfg.config['generation']['upsampling_steps'],
                                         sample=cfg.config['generation']['use_sampling'],
                                         refinement_step=cfg.config['generation']['refinement_step'],
                                         simplify_nfaces=cfg.config['generation']['simplify_nfaces'],
                                         preprocessor=None)

    def compute_loss(self, input_features_for_completion, input_points_for_completion, input_points_occ_for_completion,
                     cls_codes_for_completion, export_shape=False, weights=None):
        '''
        Compute loss for OccNet
        :param input_features_for_completion (N_B x D): Number of bounding boxes x Dimension of proposal feature.
        :param input_points_for_completion (N_B, N_P, 3): Number of bounding boxes x Number of Points x 3.
        :param input_points_occ_for_completion (N_B, N_P): Corresponding occupancy values.
        :param cls_codes_for_completion (N_B, N_C): One-hot category codes.
        :param export_shape (bool): whether to export a shape voxel example.
        :return:
        '''
        device = input_features_for_completion.device
        batch_size = input_features_for_completion.size(0)
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.to(
                device).float()
            input_features_for_completion = torch.cat(
                [input_features_for_completion, cls_codes_for_completion], dim=-1)

        kwargs = {}
        '''Infer latent code z.'''
        if self.z_dim > 0:
            q_z = self.infer_z(input_points_for_completion, input_points_occ_for_completion,
                               input_features_for_completion, device, **kwargs)
            z = q_z.rsample()
            # KL-divergence
            p0_z = self.get_prior_z(self.z_dim, device)
            kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            loss = kl
            if weights == None:
                loss = loss.mean()
            else:
                loss = (loss*weights).mean()
        else:
            z = torch.empty(size=(batch_size, 0), device=device)
            loss = 0.

        '''Decode to occupancy voxels.'''
        logits = self.decode(input_points_for_completion,
                             z, input_features_for_completion, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, input_points_occ_for_completion, reduction='none')
        loss_ce = loss_i.sum(-1)
        if weights == None:
            loss_ce = loss_ce.mean()
        else:
            loss_ce = (loss_ce*weights).mean()

        loss = loss + loss_ce

        '''Export Shape Voxels.'''
        if export_shape:
            shape = (16, 16, 16)
            p = make_3d_grid([-0.5 + 1/32] * 3, [0.5 - 1/32]
                             * 3, shape).to(device)
            p = p.expand(batch_size, *p.size())
            z = self.get_z_from_prior((batch_size,), device, sample=False)
            kwargs = {}
            p_r = self.decode(p, z, input_features_for_completion, **kwargs)

            occ_hat = p_r.probs.view(batch_size, *shape)
            voxels_out = (occ_hat >= self.threshold)
        else:
            voxels_out = None

        return loss, voxels_out

    def compute_loss_weakly_supervised(self,
                                       input_features_for_completion,
                                       object_surface_points,
                                       object_surface_normals,
                                       point_segmentation_mask,
                                       cls_codes_for_completion,
                                       knn_dict,
                                       knn_feats,
                                       export_shape=False):
        """
        Compute the loss for OccNet with weak supervision

        :param input_features_for_completion: N_B x D array (number of bboxes, dimension of proposal features)
        :param object_surface_points: N_B x N_P x 3 array (number of bboxes, number of points, XYZ)
        :param object_surface_normals: N_B x N_P x 3 array of corresponding normal vectors 
        """
        MU = 0.01  # TODO: set this property via config
        SW = 0.7 # TODO: set this property via config (sampled weight)
        KW = 0.3 # TODO: set this property via config (knn weight)
        device = input_features_for_completion.device

        # reshape points and normals to (batch_size * N_proposals x n_points x 3)

        # compute GT occupancy values based on surface noramals
        empty_points = object_surface_points + MU * object_surface_normals
        occupied_points = object_surface_points - MU * object_surface_normals
        input_points_for_completion = torch.cat(
            (empty_points, occupied_points), dim=1)

        # set occupancies and mask out background points
        empty_points_occs = torch.zeros_like(empty_points[..., 0])
        occupied_points_occs = torch.ones_like(occupied_points[..., 0]) * point_segmentation_mask.float()
        input_points_occ_for_completion = torch.cat(
            (empty_points_occs, occupied_points_occs), dim=1)

        input_points_for_completion = input_points_for_completion.to(device)
        input_points_occ_for_completion = input_points_occ_for_completion.to(device)

        sampled_loss, voxel_out = self.compute_loss(input_features_for_completion, input_points_for_completion,
                                 input_points_occ_for_completion, cls_codes_for_completion, export_shape=export_shape)
        normalizer = torch.zeros(input_points_for_completion.shape[0], dtype=torch.float32, device=device)
        knn_loss = torch.zeros(sampled_loss.shape, dtype=torch.float32, device=device)
        for i in range(knn_dict['object_encoding'].shape[1]):
            dist = (knn_dict['object_encoding'][:, i, :] - knn_feats).pow(2).sum(-1).sqrt()
            normalizer += torch.pow(dist, -1)
        normalizer = torch.pow(normalizer, -1)
        # DOESN'T WORK THAT WAY
        for i in range(knn_dict['object_encoding'].shape[1]):
            dist = (knn_dict['object_encoding'][:, i, :] - knn_feats).pow(2).sum(-1).sqrt()
            curr_loss, _ = self.compute_loss(input_features_for_completion, knn_dict['object_points'][:, i, :],
                                 knn_dict['object_points_occ'][:, i, :], cls_codes_for_completion, 
                                 export_shape=False, weights=normalizer*torch.pow(dist, -1))
            knn_loss += curr_loss
        
        return SW*sampled_loss + KW*knn_loss, voxel_out
        

    def forward(self, input_points_for_completion, input_features_for_completion, cls_codes_for_completion, sample=False, **kwargs):
        '''
        Performs a forward pass through the network.
        :param input_points_for_completion (tensor): sampled points
        :param input_features_for_completion (tensor): conditioning input
        :param cls_codes_for_completion: class codes for input shapes.
        :param sample (bool): whether to sample for z
        :param kwargs:
        :return:
        '''
        device = input_features_for_completion.device
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.to(
                device).float()
            input_features_for_completion = torch.cat(
                [input_features_for_completion, cls_codes_for_completion], dim=-1)
        '''Encode the inputs.'''
        batch_size = input_points_for_completion.size(0)
        z = self.get_z_from_prior((batch_size,), device, sample=sample)
        p_r = self.decode(input_points_for_completion, z,
                          input_features_for_completion, **kwargs)
        return p_r

    def get_z_from_prior(self, size=torch.Size([]), device='cuda', sample=False):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        p0_z = self.get_prior_z(self.z_dim, device)
        if sample:
            z = p0_z.sample(size)
        else:
            z = p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def decode(self, input_points_for_completion, z, features, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        :param input_points_for_completion: points
        :param z: latent code z
        :param features: latent conditioned features
        :return:
        '''
        logits = self.decoder(input_points_for_completion,
                              z, features, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, device, **kwargs):
        '''
        Infers latent code z.
        :param p : points tensor
        :param occ: occupancy values for occ
        :param c: latent conditioned code c
        :param kwargs:
        :return:
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(device)
            logstd_z = torch.empty(batch_size, 0).to(device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_prior_z(self, z_dim, device):
        ''' Returns prior distribution for latent code z.

        Args:
            zdim: dimension of latent code z.
            device (device): pytorch device
        '''
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

        return p0_z

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from ..utils.misc import ViewManipulator
from torch.nn.utils.rnn import pad_sequence
import math
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear, RoiQueryGenerator
from torchvision.ops import RoIAlign, nms

@HEADS.register_module()
class DVPEHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 use_polar_init=True,
                 use_angle_aug=False,
                 share_pos_embedding=False,
                 num_cut_view=6,
                 num_spin_view=3,
                 init_angle=0,
                 extra_num_group=1,
                 extra_num_query=900,
                 extra_code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
                 extra_weight=0.75,
                 loss_weight=0.75,
                 num_roi=128, 
                 num_roi_propageted=0,
                 roi_queue_len=512,
                 nms_iou_threshold=0.2,
                 in_channels=256,
                 stride=16,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 0.75,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DVPEHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.use_polar_init = use_polar_init
        self.use_angle_aug = use_angle_aug
        self.share_pos_embedding = share_pos_embedding
        self.num_cut_view = num_cut_view
        self.num_spin_view = num_spin_view
        self.init_angle = init_angle
        self.vm = ViewManipulator(num_cut_view, num_spin_view, init_angle)

        # TAG group_dn
        assert (extra_num_group == 0 and extra_num_query == 0) or (extra_num_group != 0 and extra_num_query != 0)
        self.extra_num_group = extra_num_group 
        self.extra_num_query = extra_num_query
        self.extra_code_weights = extra_code_weights
        self.extra_weight = extra_weight
        self.loss_weight = loss_weight

        self.extra_num_group_query = extra_num_group*extra_num_query
        self.all_num_query = num_query + self.extra_num_group_query



        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.LID = LID
        self.depth_start = depth_start
        self.stride=stride

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear

        # TAG roi_temp
        if num_roi == 0:
            assert num_roi_propageted == 0 and roi_queue_len == 0
        if roi_queue_len == 0:
            assert num_roi_propageted == 0 and roi_queue_len == 0
        self.num_roi = num_roi
        self.num_roi_propageted = num_roi_propageted
        self.nms_iou_threshold = nms_iou_threshold
        self.roi_queue_len = roi_queue_len
        super(DVPEHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        # TAG group_dn
        for layer in self.transformer.decoder.layers:
            layer.extra_num_group = self.extra_num_group
            layer.extra_num_query = self.extra_num_query
            layer.num_query = self.num_query

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)
        self.extra_code_weight = nn.Parameter(torch.tensor(
            self.extra_code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )

        self.memory_embed = nn.Sequential(
                nn.Linear(self.in_channels, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        # TAG group_dn
        self.reference_points = nn.Embedding(self.all_num_query, 3)
        if self.num_propagated > 0 :
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        # TAG query_pos
        if self.num_roi > 0:
            self.roi_align = RoIAlign(output_size=(7,7), spatial_scale=1/self.stride, sampling_ratio=-1)
            self.roi2query = RoiQueryGenerator(out_channels=64, rp_dim=256, num_k=9, emb_dim=self.embed_dims, num_cls=self.num_classes)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if self.share_pos_embedding:
            self.view_query_embedding = self.query_embedding
        else:
            self.view_query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        self.spatial_alignment = MLN(8)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        if self.use_polar_init:
            r = torch.sqrt(torch.rand(self.all_num_query)) * 0.5
            theta = torch.rand(self.all_num_query) * 2 * math.pi
            self.reference_points.weight.data[:, 0] = r *  torch.cos(theta)
            self.reference_points.weight.data[:, 1] = r *  torch.sin(theta)
            self.reference_points.weight.data[:, 0:2] += 0.5
            self.reference_points.weight.data[:, 2].uniform_(0,1)
        else:
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            if self.use_polar_init:
                r = torch.sqrt(torch.rand(self.num_propagated)) * 0.5
                theta = torch.rand(self.num_propagated) * 2 * math.pi
                self.pseudo_reference_points.weight.data[:, 0] = r *  torch.cos(theta)
                self.pseudo_reference_points.weight.data[:, 1] = r *  torch.sin(theta)
                self.pseudo_reference_points.weight.data[:, 0:2] += 0.5
                self.pseudo_reference_points.weight.data[:, 2].uniform_(0,1)
            else:
                nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        self.roi_queue_embedding = None
        self.roi_queue_reference_point = None
        self.roi_queue_timestamp = None
        self.roi_queue_egopose = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
        
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        # TAG group_dn
        B = rec_score.size(0)
        if self.training and self.extra_num_group > 0:
            Q = rec_score.size(1)-self.extra_num_group_query
            _, topk_indexes = torch.topk(rec_score[:, :Q], self.topk_proposals, dim=1)
        else:
            _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)

        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)


        img_points = coords3d[..., -90:-87]
        return coords_position_embeding, cone, img_points

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(inverse_sigmoid(temp_reference_point)))
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        # TAG roi_temp
        if self.num_roi > 0:
            roi_reference_point = (self.roi_queue_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
            roi_pos = self.query_embedding(pos2posemb3d(inverse_sigmoid(roi_reference_point)))
            roi_embedding = self.roi_queue_embedding

        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)
            # TAG roi_temp
            if self.num_roi > 0:
                roi_ego_motion = torch.cat([torch.zeros_like(roi_reference_point[...,:2]), self.roi_queue_timestamp, 
                                            self.roi_queue_egopose[..., :3, :].flatten(-2)], dim=-1).float()
                roi_ego_motion = nerf_positional_encoding(roi_ego_motion)
                roi_embedding = self.ego_pose_memory(roi_embedding, roi_ego_motion)
                roi_pos = self.ego_pose_pe(roi_pos, roi_ego_motion)


        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())
        if self.num_roi > 0:
            roi_pos += self.time_embedding(pos2posemb1d(self.roi_queue_timestamp).float())

        if self.num_propagated > 0:
            E = self.extra_num_group_query if self.training else 0
            Q = tgt.size(1)-E
            tgt = torch.cat([tgt[:, :Q], temp_memory[:, :self.num_propagated], tgt[:, Q:]], dim=1)
            query_pos = torch.cat([query_pos[:, :Q], temp_pos[:, :self.num_propagated], query_pos[:, Q:]], dim=1)
            reference_points = torch.cat([reference_points[:, :Q], temp_reference_point[:, :self.num_propagated], reference_points[:, Q:]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
        
        if self.num_roi_propageted > 0:
            E = self.extra_num_group_query if self.training else 0
            Q = tgt.size(1)-E
            tgt = torch.cat([tgt[:, :Q], roi_embedding[:, :self.num_roi_propageted], tgt[:, Q:]], dim=1)
            query_pos = torch.cat([query_pos[:, :Q], roi_pos[:, :self.num_roi_propageted], query_pos[:, Q:]], dim=1)
            reference_points = torch.cat([reference_points[:, :Q], roi_reference_point[:, :self.num_roi_propageted], reference_points[:, Q:]], dim=1)
            roi_embedding = roi_embedding[:, self.num_roi_propageted:]
            roi_pos = roi_pos[:, self.num_roi_propageted:]
        if self.num_roi > 0:
            temp_memory = torch.cat([temp_memory, roi_embedding[:, self.num_roi-self.num_roi_propageted:]], dim=1)
            temp_pos = torch.cat([temp_pos, roi_pos[:, self.num_roi-self.num_roi_propageted:]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1), reference_points], dim=1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling and roi_temp
            query_size = pad_size + self.num_query + self.num_propagated + self.num_roi_propageted
            tgt_size = pad_size + self.num_query + self.memory_len + self.roi_queue_len + self.num_roi_propageted
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict
    
    def get_image_points(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        # pos_embed  = inverse_sigmoid(coords3d)
        # coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic_for_cone = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        # cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        img_points = coords3d[..., -3:]
        return coords3d, intrinsic_for_cone, img_points

    
    def prepare_for_view(self, data, memory_center, topk_indexes, img_metas, reference_points, memory):
        coords3d, intrinsic_for_cone, img_points = self.get_image_points(data, memory_center, topk_indexes, img_metas)
        B = reference_points.size(0)
        V = self.num_cut_view
        S = self.num_spin_view
        if self.training and self.use_angle_aug:
            self.vm.angle_aug()
        view_dicts = []
        for state in range(S):
            query_indices, query_lens, restore_indices = \
                self.vm.cut_batch_view(reference_points.detach()-0.5, state, restore=True)
            key_indices, key_lens = \
                self.vm.cut_batch_view(img_points.detach()-0.5, state)
            key_maxlen = max(key_lens)
            view_dict = dict(
                query_indices=query_indices,
                query_lens=query_lens,
                restore_indices=restore_indices,
                key_indices=key_indices,
                key_lens=key_lens,
            )
            reference_points_list = []
            key_list = []
            coords3d_list = []
            intrinsic_for_cone_list = []
            view_key_padding_mask = torch.zeros(B*V, key_maxlen, dtype=torch.bool, device=reference_points.device)
            for b in range(B):
                for v in range(V):
                    qidx = query_indices[b*V + v]
                    kidx = key_indices[b*V + v]
                    klen = key_lens[b*V + v]
                    # self.view_query_embedding(pos2posemb3d(reference_points))
                    rp = self.vm.transform_to_view(reference_points[b, qidx], v, state)
                    reference_points_list.append(rp)
                    key_list.append(memory[b, kidx])
                    intrinsic_for_cone_list.append(intrinsic_for_cone[b, kidx])
                    coord = coords3d[b, kidx].reshape(klen, -1, 3)
                    coord = self.vm.transform_to_view(coord, v, state).reshape(klen, -1)
                    coords3d_list.append(coord)
                    view_key_padding_mask[b*V+v,  klen:] = True
                    
            view_reference_points = pad_sequence(reference_points_list, batch_first=False)
            view_key = pad_sequence(key_list, batch_first=False)
            view_coords3d = pad_sequence(coords3d_list, batch_first=False)
            view_intrinsic_for_cone = pad_sequence(intrinsic_for_cone_list, batch_first=False)

            

            # PE
            view_query_pos = self.view_query_embedding(pos2posemb3d(inverse_sigmoid(view_reference_points)))
            pos_embed  = inverse_sigmoid(view_coords3d)
            view_key_pos = self.position_encoder(pos_embed)

            #  spatial_alignment in focal petr
            view_cone = torch.cat([view_intrinsic_for_cone, view_coords3d[..., -3:], view_coords3d[..., -90:-87]], dim=-1) 
            view_key = self.spatial_alignment(view_key, view_cone)
            view_key_pos = self.featurized_pe(view_key_pos, view_key)


            view_dict.update(view_query_pos=view_query_pos,
                             view_key=view_key, view_key_pos=view_key_pos,
                             view_key_padding_mask=view_key_padding_mask)
            view_dicts.append(view_dict)
        return view_dicts


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DVPEHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    

    # TAG: query_emb init target
    def get_current_roi(self, outs_roi, data):
        img_feats = data['img_feats']
        B, N, C, H, W = img_feats.shape
        img_feats = img_feats.flatten(end_dim=1)
        img_h, img_w = data['img'][0].shape[-2:]


        # roi feature
        bbox_preds = outs_roi['enc_bbox_preds']
        bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds)
        bbox_preds = bbox_preds.reshape(B*N, -1, 4)
        cls_scores = outs_roi['enc_cls_scores']
        cls_scores = cls_scores.reshape(B*N, -1, self.num_classes)
        cls_max_score = cls_scores.max(dim=-1)[0]
        #pred_depth2d = outs_roi['pred_depth2d'].exp()

        # self.save_2d(data,bbox_preds, cls_scores)
        # nms & select topk bbox
        cls_scores_list = []
        bbox_preds_list = []
        bn_indices_list = []
        depth_preds_list =[]
        for b_idx in range(B):
            clses_n_list = []
            bbox_n_list = []
            cls_n_list = []
            bn_n_list = []
            depth_n_list = []
            for n_idx in range(N):
                bn_idx = b_idx * N + n_idx
                bbox = bbox_preds[bn_idx]
                cls = cls_max_score[bn_idx]
                keep = nms(bbox, cls, self.nms_iou_threshold)
                if keep.size(0) < self.num_roi / N:
                    keep = nms(bbox, cls, 0.8)
                bbox = bbox[keep]
                cls = cls[keep]
                bn_indices = torch.full_like(cls, bn_idx, dtype=torch.int64)
                clses_n_list.append(cls_scores[bn_idx][keep])
                bbox_n_list.append(bbox)
                cls_n_list.append(cls)
                bn_n_list.append(bn_indices)
                #depth_n_list.append(pred_depth2d[bn_idx][keep])
            clses_n = torch.cat(clses_n_list, dim=0)
            bbox_n = torch.cat(bbox_n_list, dim=0)
            cls_n = torch.cat(cls_n_list, dim=0)
            bn_n = torch.cat(bn_n_list, dim=0)
            #depth_n = torch.cat(depth_n_list, dim=0)
            assert cls_n.size(0) >= self.num_roi
            scores, indices = cls_n.topk(self.num_roi, dim=0)
            cls_scores_list.append(clses_n[indices])
            bbox_preds_list.append(bbox_n[indices])
            bn_indices_list.append(bn_n[indices])
            #depth_preds_list.append(depth_n[indices])

        clses = torch.cat(cls_scores_list, dim = 0)
        bboxes = torch.cat(bbox_preds_list, dim = 0)
        bn_indices = torch.cat(bn_indices_list, dim = 0)
        #depths = torch.cat(depth_preds_list, dim = 0)
        factors = bboxes.new_tensor([img_w,img_h,img_w,img_h]).unsqueeze(0)
        bboxes = factors * bboxes
        rois = torch.cat([bn_indices.unsqueeze(-1), bboxes], dim=-1)
        rois_feature = self.roi_align(img_feats, rois)
        

        # img2lidars = data['lidar2img'].inverse().reshape(-1,4,4)[bn_indices]
        # x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2 * depths
        # y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2 * depths
        # roi_points = torch.stack([x_center, y_center, depths, torch.ones_like(depths)], dim=-1)
        # tgt_emb = self.roic2emb(rois_feature, clses)
        ## caculate roi intrinsics : cam->roi
        # tgt_emb = tgt_emb.reshape(B, self.num_roi, -1)
        # tgt_emb = tgt_emb.reshape(B, self.num_roi, -1)

        # roi_reference_points = img2lidars @ roi_points.unsqueeze(-1)
        # roi_reference_points = roi_reference_points.squeeze(-1)[:, :3].reshape(B, self.num_roi, 3)
        # roi_reference_points = (roi_reference_points - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

        # reference_points = torch.cat([roi_reference_points, 
        #                               self.reference_points.weight.unsqueeze(0).repeat(B, 1, 1)],
        #                              dim=1)

        intrinsics = data['intrinsics'].detach().flatten(end_dim=1)[bn_indices]
        extrinsics = data['extrinsics'].detach().flatten(end_dim=1)[bn_indices]
        x_max = rois[:, 3]
        x_min = rois[:, 1]
        y_max = rois[:, 4]
        y_min = rois[:, 2]
        H_roi, W_roi= self.roi_align.output_size
        r_x = W_roi/(x_max - x_min)
        r_y = H_roi/(y_max - y_min)
        roi_intrinsics = intrinsics.new_zeros((4,4)).unsqueeze(0).repeat(B*self.num_roi, 1, 1)
        roi_intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * r_x
        roi_intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * r_y
        roi_intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - x_min) * r_x
        roi_intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - y_min) * r_y
        roi_intrinsics[:, 2, 2] = 1.0
        roi_intrinsics[:, 3, 3] = 1.0
        roi_emb, roi_reference_points = self.roi2query(rois_feature, roi_intrinsics[:, :3, :3], clses)
        roi_emb = roi_emb.reshape(B, self.num_roi, -1)
        
        roi_reference_points = self.roi2query.decode(roi_reference_points)
    
  
        roi_reference_points = torch.cat([roi_reference_points, roi_reference_points.new_ones((B*self.num_roi, 1))], dim=-1)
        roi_reference_points =  torch.inverse(extrinsics) @ torch.inverse(roi_intrinsics) @ roi_reference_points.unsqueeze(-1)
        roi_reference_points = roi_reference_points.squeeze(-1)[:, :3].reshape(B, self.num_roi, 3)
        # roi_reference_points = (roi_reference_points - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        # reference_points = torch.cat([roi_reference_points, 
        #                               self.reference_points.weight.unsqueeze(0).repeat(B, 1, 1)],
        #                               dim=1)
        # reference_points = self.reference_points.weight.unsqueeze(0).repeat(B, 1, 1)
        

        return  roi_emb, roi_reference_points


    def pre_update_roi_queue(self, outs_roi, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        roi_queue_len = self.roi_queue_len + self.num_roi
        if self.roi_queue_embedding is None:
            self.roi_queue_embedding = x.new_zeros(B, roi_queue_len, self.embed_dims)
            self.roi_queue_reference_point = x.new_zeros(B, roi_queue_len, 3)
            self.roi_queue_timestamp = x.new_zeros(B, roi_queue_len, 1)
            self.roi_queue_egopose = x.new_zeros(B, roi_queue_len, 4, 4)
        else:
            self.roi_queue_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.roi_queue_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.roi_queue_egopose
            self.roi_queue_reference_point = transform_reference_points(self.roi_queue_reference_point, data['ego_pose_inv'], reverse=False)
            self.roi_queue_embedding = memory_refresh(self.roi_queue_embedding.detach(), x)
            self.roi_queue_reference_point = memory_refresh(self.roi_queue_reference_point.detach(), x)
            self.roi_queue_timestamp = memory_refresh(self.roi_queue_timestamp, x)
            self.roi_queue_egopose = memory_refresh(self.roi_queue_egopose, x)
        
        rec_roi_embedding, rec_roi_reference_points = self.get_current_roi(outs_roi, data)
        rec_roi_timestamp = x.new_zeros(B, self.num_roi, 1, dtype=torch.float64)
        rec_roi_egopose = torch.eye(4, device=x.device).unsqueeze(0).unsqueeze(0).repeat(B, self.num_roi, 1, 1)
        self.roi_queue_embedding = torch.cat([rec_roi_embedding, self.roi_queue_embedding], dim=1)[:, :roi_queue_len]
        self.roi_queue_reference_point = torch.cat([rec_roi_reference_points, self.roi_queue_reference_point], dim=1)[:, :roi_queue_len]
        self.roi_queue_timestamp = torch.cat([rec_roi_timestamp, self.roi_queue_timestamp], dim=1)[:, :roi_queue_len]
        self.roi_queue_egopose = torch.cat([rec_roi_egopose, self.roi_queue_egopose], dim=1)[:, :roi_queue_len]

    def post_update_roi_queue(self, data):
        self.roi_queue_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.roi_queue_reference_point = transform_reference_points(self.roi_queue_reference_point, data['ego_pose'], reverse=False)
        self.roi_queue_egopose = data['ego_pose'].unsqueeze(1) @ self.roi_queue_egopose
        
    def forward(self, memory_center, img_metas, topk_indexes=None, outs_roi=None,  **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # zero init the memory bank
        self.pre_update_memory(data)
        # TAG roi_temp
        if self.num_roi > 0:
            self.pre_update_roi_queue(outs_roi, data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = topk_gather(memory, topk_indexes)

        # pos_embed, cone, img_points = self.position_embeding(data, memory_center, topk_indexes, img_metas)

        memory = self.memory_embed(memory)

        # # spatial_alignment in focal petr
        # memory = self.spatial_alignment(memory, cone)
        # pos_embed = self.featurized_pe(pos_embed, memory)

        # inital reference_points
        if self.training:
            reference_points = self.reference_points.weight
        else:
            reference_points = self.reference_points.weight[:self.num_query]
        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1)
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        query_pos = self.query_embedding(pos2posemb3d(inverse_sigmoid(reference_points)))
        # inital query
        tgt = torch.zeros_like(query_pos)
        # prepare for the tgt and query_pos using mln.
        # TODO
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)

        view_dicts = self.prepare_for_view(data, memory_center, topk_indexes, img_metas, reference_points, memory)
        # transformer here is a little different from PETR
        pos_embed = None
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_embed, 
                                       attn_mask, temp_memory, temp_pos, view_dicts=view_dicts)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        V = self.num_cut_view
        S = self.num_spin_view
        for lvl in range(outs_dec.shape[0]):
            state = lvl%S
            if outs_dec.shape[0] == 1: # return_intermediate = False
                state = S-1
            view_dict = view_dicts[state]
            query_indices = view_dict['query_indices']
            
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            for b in range(B):
                for v in range(V):
                    qidx = query_indices[b*V + v]
                    tmp[b, qidx, 0:2] = self.vm.transform_to_view(tmp[b, qidx, 0:2], v, state, inverse=True, trans=False, dim=2)
                    tmp[b, qidx, 8:10] = self.vm.transform_to_view(tmp[b, qidx, 8:10], v, state, inverse=True, trans=False, dim=2)
                    tmp[b, qidx, 6:8] = self.vm.transform_to_view(tmp[b, qidx, 6:8], v, state, inverse=True, trans=False, dim=1)
            
            bbox = tmp
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            bbox[..., 0:3] += reference[..., 0:3]
            bbox[..., 0:3] = bbox[..., 0:3].sigmoid()
            
            outputs_coord = bbox
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)
        if self.num_roi > 0:
            self.post_update_roi_queue(data)
        outs = dict()
        if self.training:
            if mask_dict and mask_dict['pad_size'] > 0:
                output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
                output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
                all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
                all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
                outs["dn_mask_dict"] = mask_dict
            else:
                outs["dn_mask_dict"] = None

            if self.extra_num_group > 0:
                D, B = all_cls_scores.shape[:2]
                G = self.extra_num_group
                Q = self.num_query + self.num_propagated + self.num_roi_propageted
                E = self.extra_num_query
                extra_cls_scores = all_cls_scores[:, :, Q:, :].reshape(D, B*G, E, -1)
                extra_bbox_preds = all_bbox_preds[:, :, Q:, :].reshape(D, B*G, E, -1)
                all_cls_scores = all_cls_scores[:, :, :Q, :]
                all_bbox_preds = all_bbox_preds[:, :, :Q, :]
                outs["extra_cls_scores"] = extra_cls_scores
                outs["extra_bbox_preds"] = extra_bbox_preds
            else:
                outs["extra_cls_scores"] = None
                outs["extra_bbox_preds"] = None
        outs.update({
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
        })
        return outs
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls*self.loss_weight, loss_bbox*self.loss_weight

    def extra_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.extra_code_weight

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls*self.extra_weight, loss_bbox*self.extra_weight
    
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)
        
        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        # TAG group_dn extra_loss
        extra_gt_bboxes_list = [gt for gt in gt_bboxes_list for _ in range(self.extra_num_group)]
        extra_gt_labels_list = [gt for gt in gt_labels_list for _ in range(self.extra_num_group)]
        extra_all_gt_bboxes_list = [extra_gt_bboxes_list for _ in range(num_dec_layers)]
        extra_all_gt_labels_list = [extra_gt_labels_list for _ in range(num_dec_layers)]
        if preds_dicts['extra_cls_scores'] is not None:
            extra_cls_scores = preds_dicts['extra_cls_scores']
            extra_bbox_preds = preds_dicts['extra_bbox_preds']
            losses_extra_cls, losses_extra_bbox = multi_apply(
                self.extra_loss_single, extra_cls_scores, extra_bbox_preds,
                extra_all_gt_bboxes_list, extra_all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['loss_extra_cls'] = losses_extra_cls[-1]
            loss_dict['loss_extra_bbox'] = losses_extra_bbox[-1]
            num_dec_layer = 0
            for loss_extra_cls_i, loss_extra_bbox_i in zip(losses_extra_cls[:-1],
                                            losses_extra_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_extra_cls'] = loss_extra_cls_i
                loss_dict[f'd{num_dec_layer}.loss_extra_bbox'] = loss_extra_bbox_i
                num_dec_layer += 1


        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

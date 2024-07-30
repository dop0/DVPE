import torch
import torch.nn as nn
import numpy as np
from mmdet.core import bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid
import math

def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def apply_ltrb(locations, pred_ltrb): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)


        return pred_boxes    

def apply_center_offset(locations, center_offset): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations



def gaussian_2d(shape, sigma=1.0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
        

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

class ViewManipulator:
    def __init__(self, num_view, num_spin, init_angle=0):
        self.num_view = num_view
        self.num_spin = num_spin
        self.init_angle = math.radians(init_angle)
        self.aug_angle = 0
        self.fov = 2*math.pi / self.num_view
        self.state_angle = self.fov / self.num_spin
        
    
    def angle_aug(self):
        self.aug_angle = torch.rand(1).item() * self.fov

    def _cut_by_angle(self, coordinates, state):
        angles = torch.atan2(coordinates[:, :, 1], coordinates[:, :, 0])
        rotation_angle = self.init_angle + self.aug_angle + self.state_angle * state
        angles = torch.fmod(angles + rotation_angle, 2 * math.pi)
        groups = torch.floor(angles / self.fov).long() % self.num_view
        return groups

    def cut_batch_view(self, coordinates, state, restore=False):
        '''
        points : (B, N, 3)
        n : int
        '''
        B = coordinates.shape[0]
        groups = self._cut_by_angle(coordinates, state)
        lens = []
        indices = []
        for b in range(B):
            for v in range(self.num_view):
                index = torch.nonzero(groups[b] == v, as_tuple=False).flatten()
                indices.append(index)
                lens.append(index.size(0))
            
        if not restore:
            return indices, lens

        restore_indices = []
        for b in range(B):
            batch_index = torch.cat(indices[b*self.num_view:(b+1)*self.num_view])
            restore_indices.append(torch.argsort(batch_index))
        return indices, lens, restore_indices

    def transform_to_view(self, coords, view_idx, state, inverse=False, trans=True, dim=3):
        rotation_angle = self.init_angle + self.aug_angle + self.state_angle * state
        angle = self.fov*(2*view_idx+1) - 2*rotation_angle
        angle = (math.pi - angle)/2
        if not inverse:
            angle = -angle
        if dim == 3:
            R = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                            [math.sin(angle), math.cos(angle), 0],
                            [0, 0, 1]]).to(coords.device)
        elif dim == 2:
            R = torch.tensor([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]]).to(coords.device)
        elif dim == 1:
            R = torch.tensor([[math.cos(angle), math.sin(angle)],
                            [-math.sin(angle), math.cos(angle)]]).to(coords.device)
        if trans:
            return (coords-0.5)@R + 0.5
        else:
            return coords@R
    # TAG: query_pos
class ReferencePointsGenerater(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, out_channels=64, mlp_dim1=256, mlp_dim2=64, num_k=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 3 * 3 + num_k, mlp_dim1),
            nn.ReLU(),
            nn.Linear(mlp_dim1, mlp_dim2),
            nn.ReLU(),
            nn.Linear(mlp_dim2, 3)
        )


    def forward(self, x, k):
        x = self.conv(x)
        x = torch.cat([x.reshape(x.size(0), -1), k.reshape(k.size(0), -1)], dim = 1)
        x = self.mlp(x)
        return x
    
    # TAG: roi_temp
class RoiQueryGenerator(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, out_channels=64, rp_dim=256, num_k=9, emb_dim=256, num_cls=10, roi_range=[0.0, 0, 0, 7, 7, 51.2]):
        super().__init__()
        self.roi_range = nn.Parameter(torch.tensor(roi_range), requires_grad=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rp_mlp = nn.Sequential(
            nn.Linear(out_channels * 3 * 3 + num_k, rp_dim),
            nn.ReLU(),
            nn.Linear(rp_dim, 3)
        )
        self.emb_mlp = nn.Sequential(
            nn.Linear(out_channels * 3 * 3 + num_cls, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x, k, c):
        x = self.conv(x)
        xk = torch.cat([x.reshape(x.size(0), -1), k.reshape(k.size(0), -1)], dim = 1)
        roi_rp = self.rp_mlp(xk)
        xc = torch.cat([x.reshape(x.size(0), -1), c], dim = 1)
        roi_emb = self.emb_mlp(xc)
        return roi_emb, roi_rp
    
    def decode(self, rp):
        rp = rp.sigmoid() * (self.roi_range[3:6] - self.roi_range[0:3]) + self.roi_range[0:3]
        return rp
        

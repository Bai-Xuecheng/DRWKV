# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

# other import
import os
import math
import numpy as np
from math import exp
from torchvision import models, transforms

try:
    from pytorch_ssim import _ssim, create_window
except ModuleNotFoundError:
    _ssim = None
    create_window = None


# class VGGLoss(nn.Module):
#     def __init__(self, conv_index='54', rgb_range=1):
#         super(VGGLoss, self).__init__()
#         vgg_features = models.vgg19(pretrained=True).features
#         modules = [m for m in vgg_features]
#         if conv_index == '22':
#             self.vgg = nn.Sequential(*modules[:8])
#             self.vgg.cuda()
#         elif conv_index == '54':
#             self.vgg = nn.Sequential(*modules[:35])
#             self.vgg.cuda()

#         vgg_mean = (0.485, 0.456, 0.406)
#         vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
#         self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std).cuda()
#         self.vgg.requires_grad = False

#     def forward(self, sr, hr):
#         def _forward(x):
#             x = self.sub_mean(x)
#             x = self.vgg(x)
#             return x
            
#         vgg_sr = _forward(sr)
#         with torch.no_grad():
#             vgg_hr = _forward(hr.detach())

#         loss = F.mse_loss(vgg_sr, vgg_hr)

#         return loss
class VGGLoss(nn.Module):
    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, device, model='vgg19', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.device = device
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1].to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)

class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        if create_window is None:
            raise ModuleNotFoundError("pytorch_ssim is required to use SSIM_loss")
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def _spatial_gradients(x):
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x, grad_y


def _total_variation(x):
    grad_x, grad_y = _spatial_gradients(x)
    return grad_x.abs().mean() + grad_y.abs().mean()


class MS2Loss(nn.Module):
    """Multi-Structure Spectral Smoothness Loss from the DRWKV paper.

    The model output is expected to be a dictionary containing the enhanced
    image plus GER auxiliary maps: edge, illumination, artifact, and the
    alpha/beta/gamma regularization parameters.
    """

    def __init__(
        self,
        lambda_recon=1.0,
        lambda_sparse=0.01,
        lambda_smooth=0.1,
        lambda_artifact=0.05,
        lambda_reg=1e-4,
        edge_aware_weight=10.0,
        artifact_tv_weight=0.1,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.lambda_artifact = lambda_artifact
        self.lambda_reg = lambda_reg
        self.edge_aware_weight = edge_aware_weight
        self.artifact_tv_weight = artifact_tv_weight
        self.reconstruction = nn.L1Loss()

    def _illumination_smoothness(self, illumination, low_light):
        illum_dx, illum_dy = _spatial_gradients(illumination)
        image_dx, image_dy = _spatial_gradients(low_light.mean(dim=1, keepdim=True))

        smooth_x = illum_dx.abs() * torch.exp(-self.edge_aware_weight * image_dx.abs())
        smooth_y = illum_dy.abs() * torch.exp(-self.edge_aware_weight * image_dy.abs())
        return smooth_x.mean() + smooth_y.mean()

    def _regularization(self, output):
        params = []
        for name in ("alpha", "beta", "gamma"):
            value = output.get(name)
            if value is not None:
                params.append(value.pow(2).mean())
        if not params:
            enhanced = output["enhanced"]
            return enhanced.new_tensor(0.0)
        return torch.stack(params).sum()

    def forward(self, output, target, low_light):
        if not isinstance(output, dict):
            output = {"enhanced": output}

        enhanced = output["enhanced"]
        edge = output.get("edge", enhanced.new_zeros(enhanced.shape[0], 1, enhanced.shape[2], enhanced.shape[3]))
        illumination = output.get("illumination", low_light.mean(dim=1, keepdim=True))
        artifact = output.get("artifact", torch.zeros_like(enhanced))

        loss_recon = self.reconstruction(enhanced, target)
        loss_sparse = edge.abs().mean()
        loss_smooth = self._illumination_smoothness(illumination, low_light)
        loss_artifact = artifact.abs().mean() + self.artifact_tv_weight * _total_variation(artifact)
        loss_reg = self._regularization(output)

        total = (
            self.lambda_recon * loss_recon
            + self.lambda_sparse * loss_sparse
            + self.lambda_smooth * loss_smooth
            + self.lambda_artifact * loss_artifact
            + self.lambda_reg * loss_reg
        )
        components = {
            "loss/recon": loss_recon.detach(),
            "loss/sparse": loss_sparse.detach(),
            "loss/smooth": loss_smooth.detach(),
            "loss/artifact": loss_artifact.detach(),
            "loss/reg": loss_reg.detach(),
            "loss/total": total.detach(),
        }
        return total, components
 
# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)


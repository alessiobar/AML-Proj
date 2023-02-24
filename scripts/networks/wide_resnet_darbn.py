import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import pickle

import sys
import numpy as np

# From Appendix E in Zada et al., 2022

def dar_bn(bn_layer, x, noise_mask):
    """Applies DAR-BN normalization to a 4D input (a mini-batch of 2D inputs with
        additional channel dimension)

    bn_layer : torch.nn.BatchNorm2d
        Batch norm layer operating on activation maps of natural images
    x : torch.FloatTensor of size: (N, C, H, W)
        2D activation maps obtained from both natural images and noise images
    noise_mask: torch.BoolTensor of size: (N)
        Boolean 1D tensor indicates which activation map is obtained from noise
    """
    # Batch norm for activation maps of natural images
    out_natural = bn_layer(x[torch.logical_not(noise_mask)])
    # Batch norm for activation maps of noise images
    # Do not compute gradients for this operation
    with torch.no_grad():
        adaptive_params = {"weight": bn_layer.weight, "bias": bn_layer.bias,
                            "eps": bn_layer.eps}
        out_noise = batch_norm_with_adaptive_parameters(x[noise_mask],
                    adaptive_params)

    # Concatenate activation maps in original order
    out = torch.empty_like(torch.cat([out_natural, out_noise], dim=0))
    out[torch.logical_not(noise_mask)] = out_natural
    out[noise_mask] = out_noise

    return out

def batch_norm_with_adaptive_parameters(x_noise, adaptive_parameters):
    """Applies batch normalization to x_noise according to adaptive_parameters

    x_noise : torch.FloatTensor of size: (N, C, H, W)
        2D activation maps obtained from noise images only
    adaptive_parameters:
        a dictionary containing:
            weight: scale parameter for the adaptive affine
            bias: bias parameter for the adaptive affine
            eps: a value added to the denominator for numerical stability.
    """
    # Calculate mean and variance for the noise activations batch per channel
    mean = x_noise.mean([0, 2, 3])
    var = x_noise.var([0, 2, 3], unbiased=False)
    # Normalize the noise activations batch per channel
    out = x_noise - mean[None, :, None, None]
    out = out / torch.sqrt(var[None, :, None, None] + adaptive_parameters["eps"])

    # Scale and shift using adaptive affine per channel
    out = out * adaptive_parameters["weight"][None, :, None, None] + adaptive_parameters["bias"][None, :, None, None]

    return out

# Adapted from https://github.com/meliketoy/wide-resnet.pytorch, PyTorch implementation of
# Zagoruyko, S., & Komodakis, N. (2016). Wide residual networks. arXiv preprint arXiv:1605.07146.

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        with open("noise_mask.pkl", "rb") as f:
            noise_mask = pickle.load(f)
        out = self.dropout(self.conv1(F.relu(dar_bn(self.bn1, x, noise_mask))))
        out = self.conv2(F.relu(dar_bn(self.bn2, out, noise_mask)))
        out += self.shortcut(x)

        return out

class Wide_ResNet_DarBN(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_DarBN, self).__init__()
        self.in_planes = 16
        self.noise_mask = None

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, noise_mask):
        with open("noise_mask.pkl", "wb") as f:
            pickle.dump(noise_mask, f)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(dar_bn(self.bn1, out, noise_mask))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet_DarBN(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())

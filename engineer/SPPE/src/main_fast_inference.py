import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
from engineer.SPPE.src.utils.img import flip, shuffleLR
from engineer.SPPE.src.utils.eval import getPrediction
from engineer.SPPE.src.models.FastPose import createModel
from engineer.SPPE.src.models.pose_hrnet_Simcc import createHRModel
import visdom
import time
import sys

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet, self).__init__()

        model = createModel().cuda()
        print('Loading pose model from {}'.format('./models/sppe/duc_se.pth'))
        sys.stdout.flush()
        model.load_state_dict(torch.load('./models/sppe/duc_se.pth'))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        flip_out = self.pyranet(flip(x))
        flip_out = flip_out.narrow(1, 0, 17)

        flip_out = flip(shuffleLR(
            flip_out, self.dataset))

        out = (flip_out + out) / 2

        return out


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet_fast, self).__init__()

        model = createModel().cuda()
        print('Loading pose model from {}'.format('./weights/sppe/duc_se.pth'))
        model.load_state_dict(torch.load('/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/weights/sppe/duc_se.pth'))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out,ret_features = self.pyranet(x)
        out = out.narrow(1, 0, 17)


        return out,ret_features

class InferenNet_hr(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet_hr, self).__init__()

        model = createHRModel().cuda()
        print('Loading pose model from {}'.format('./weights/hrnet/pose_hrnet_w32_256*160.pth'))
        model.load_state_dict(torch.load('/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/weights/hrnet/pose_hrnet_w32_Simcc.pth'))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out,ret_features = self.pyranet(x)
        out = out.narrow(1, 0, 17)
        ret = []

# w32
        channel_increase_layer_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1)
        channel_increase_layer_1 = channel_increase_layer_1.to('cuda')
        channel_increase_layer_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
        channel_increase_layer_2 = channel_increase_layer_2.to('cuda')
        channel_increase_layer_3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)
        channel_increase_layer_3 = channel_increase_layer_3.to('cuda')
        ret.append(channel_increase_layer_3(ret_features[0]))
        ret.append(channel_increase_layer_2(ret_features[1]))
        ret.append(channel_increase_layer_1(ret_features[2]))
# w32 wanzheng
#         channel_increase_layer_1 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=1)
#         channel_increase_layer_1 = channel_increase_layer_1.to('cuda')
#         channel_increase_layer_2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=1)
#         channel_increase_layer_2 = channel_increase_layer_2.to('cuda')
#         channel_increase_layer_3 = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1)
#         channel_increase_layer_3 = channel_increase_layer_3.to('cuda')
#         ret.append(channel_increase_layer_3(ret_features[0]))
#         ret.append(channel_increase_layer_2(ret_features[1]))
#         ret.append(channel_increase_layer_1(ret_features[2]))

        return out,ret

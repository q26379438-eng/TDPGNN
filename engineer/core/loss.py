# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


import torch
import torch.nn as nn

class JointConstraintLoss(nn.Module):
    def __init__(self, edges, lambda_weights):
        super(JointConstraintLoss, self).__init__()
        self.edges = edges  # List of tuples indicating keypoint connections
        self.lambda_weights = lambda_weights  # Dictionary with edge tuples as keys and lambda values as values

    def forward(self, predicted, target):
        loss = 0
        for (k1, k2) in self.edges:
            # Extract the predicted and target keypoints
            pred_k1, pred_k2 = predicted[:, k1, :], predicted[:, k2, :]
            targ_k1, targ_k2 = target[:, k1, :], target[:, k2, :]

            # Calculate the vectors
            pred_vec = pred_k1 - pred_k2
            targ_vec = targ_k1 - targ_k2

            # Calculate the distance constraint
            distance_constraint = torch.norm(pred_vec - targ_vec, dim=1).pow(2)

            # Calculate the cosine similarity (direction constraint)
            pred_norm = torch.norm(pred_vec, dim=1)
            targ_norm = torch.norm(targ_vec, dim=1)
            cosine_similarity = torch.sum(pred_vec * targ_vec, dim=1) / (pred_norm * targ_norm)
            direction_constraint = 1 - cosine_similarity

            # Apply the lambda weight for this pair of keypoints
            lambda_weight = self.lambda_weights[(k1, k2)]
            pair_loss = lambda_weight * distance_constraint + direction_constraint

            # Accumulate the loss
            loss += pair_loss.mean()

        return loss

# Example usage:
edges = [(0, 1), (1, 2), (2, 3)]  # Define keypoint connections
lambda_weights = {(0, 1): 0.5, (1, 2): 0.8, (2, 3): 0.6}  # Define lambda weights for each connection

# Instantiate the loss function
joint_constraint_loss = JointConstraintLoss(edges, lambda_weights)

# Dummy predicted and target tensors
predicted = torch.randn(8, 17, 2)  # Batch size of 8, 17 keypoints, 2D coordinates
target = torch.randn(8, 17, 2)  # Same shape as predicted

# Calculate the loss
loss = joint_constraint_loss(predicted, target)
print("Joint Constraint Loss:", loss.item())

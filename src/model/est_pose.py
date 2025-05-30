from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..config import Config

from ..utils import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d

def geodesic_loss(R_pred, R_gt,alpha = 0.1):
    trace = torch.sum(R_pred * R_gt, dim=(-2, -1))
    acos_input = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7) 
    theta = torch.acos(acos_input)
    return alpha * theta.mean()

def ortho6d_loss(ortho6d_pred, ortho6d_gt):
    pred_1,pred_2 = ortho6d_pred[:, :3], ortho6d_pred[:, 3:6]
    gt_1,gt_2 = ortho6d_gt[:, :3], ortho6d_gt[:, 3:6]

    return F.mse_loss(pred_1, gt_1) + F.mse_loss(pred_2, gt_2)


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, D) -> (B, D, N)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # Max pooling over N dimension
        x = torch.max(x, 2)[0] # (B, output_dim)
        return x
class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        self.point_encoder = PointNetEncoder(input_dim=3, output_dim=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        self.fc_6drot = nn.Linear(256, 6)
        self.trans = nn.Linear(256, 3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        b = pc.shape[0]

        global_feature = self.point_encoder(pc)

        fc1_out = F.relu(self.bn1(self.fc1(global_feature)))
        fc2_out = F.relu(self.bn2(self.fc2(fc1_out)))

        # fc2_out = self.dropout(fc2_out) We can't use this!!!

        ortho6d_pred = self.fc_6drot(fc2_out)
        trans_pred = self.trans(fc2_out)
        
        rot_pred = compute_rotation_matrix_from_ortho6d(ortho6d_pred)

        ortho6d_gt = compute_ortho6d_from_rotation_matrix(rot)

        loss_trans = F.mse_loss(trans, trans_pred)

        loss_rot = geodesic_loss(rot_pred, rot) + ortho6d_loss(ortho6d_pred, ortho6d_gt)

        loss = 10*torch.sqrt(loss_trans)+ loss_rot
        metric = dict(
            loss=loss,
            # additional metrics you want to log
            loss_trans = loss_trans,
            loss_rot = loss_rot,
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        # raise NotImplementedError("You need to implement the est function")
        self.eval()
        with torch.no_grad():
            global_feature = self.point_encoder(pc)
            fc1_out = F.relu(self.bn1(self.fc1(global_feature)))
            fc2_out = F.relu(self.bn2(self.fc2(fc1_out)))

            ortho6d = self.fc_6drot(fc2_out)
            trans = self.trans(fc2_out)


            rot = compute_rotation_matrix_from_ortho6d(ortho6d) 
            # Ensure the rotation matrix is orthogonal and has determinant 1 in called function
        return trans, rot
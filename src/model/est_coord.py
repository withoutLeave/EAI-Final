from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis

from .est_pose import PointNetEncoder

def svd(src,target):
    """
    Compute the optimal rotation matrix using Singular Value Decomposition (SVD).
    """
    assert src.shape == target.shape
    assert src.shape[1] == 3

    # Center the points
    src = src - torch.mean(src, dim=0)
    target = target - torch.mean(target, dim=0)

    # Compute covariance matrix.
    # More usually for reflection
    H = torch.matmul(src.T, target)

    U,S,V = torch.linalg.svd(H)

    R = torch.matmul(V, U.T)

    # H = torch.matmul(target.T, src)

    # U, S, V = torch.svd(H)

    # R = torch.matmul(U,V.T)

    # Ensure a right-handed coordinate system
    if torch.det(R) < 0:
        V_new = V.clone()
        V_new[:, 2] *= -1
        R = torch.matmul(V_new, U.T)

    T = torch.mean(target, dim=0) - torch.matmul(R, torch.mean(src, dim=0))


    return R, T

def svd_2(src, target):
    """
    Compute the optimal rotation matrix using SVD.
    
    Args:
        src:    (N, 3) source points
        target: (N, 3) target points
    
    Returns:
        R: (3, 3) rotation matrix
        T: (3,) translation vector
    """
    assert src.dim() == 2 and target.dim() == 2, "Inputs must be 2D tensors"
    assert src.shape == target.shape, "Input shapes must match"
    assert src.size(1) == 3, "Points must be 3D"
    assert src.size(0) >= 3, "At least 3 points are required"

    # Center the points
    src_centered = src - torch.mean(src, dim=0)
    target_centered = target - torch.mean(target, dim=0)

    # Compute covariance matrix
    # H = src_centered.T @ target_centered
    H = target_centered.T @ src_centered

    # SVD
    U, S, Vh = torch.linalg.svd(H)

    V = Vh.mH

    # Rotation matrix
    # R = V @ U.T
    R = U @ Vh
    # R = U.T @ V

    # Handle reflection case
    if torch.det(R) < 0:
        Vh[:, 2] *= -1
        R = U @ Vh
        # R 

    # Translation
    # print("src", src.shape, "target", target.shape)
    T = torch.mean(target, dim=0) - (R @ torch.mean(src, dim=0))

    return R, T


def ransac(src: torch.Tensor, target: torch.Tensor, N: int = 3, max_iter: int = 500, distance_threshold: float = 0.01, min_inliers: int = 100):
    """
    RANSAC to find R, T such that target ≈ src @ R.T + T.

    Parameters
    ----------
    src : torch.Tensor (num_points, 3)
    target : torch.Tensor (num_points, 3)
    N : int, Minimum points to sample (default: 3).
    max_iter : int, Max iterations (default: 500).
    distance_threshold : float, Inlier distance threshold (default: 0.01).
    min_inliers : int, Minimum required inliers (default: 100).

    Returns
    -------
    torch.Tensor (3, 3) : Best R
    torch.Tensor (3,) : Best T
    torch.Tensor (num_points,) : Inlier mask
    """
    assert src.shape == target.shape and src.dim() == 2 and src.shape[1] == 3
    num_points = src.shape[0]
    assert num_points >= N

    best_R = torch.eye(3, device=src.device, dtype=src.dtype)
    best_T = torch.zeros(3, device=src.device, dtype=src.dtype)
    best_inlier_count = -1
    best_inliers_mask = torch.zeros(num_points, dtype=torch.bool, device=src.device)
    best_R_candidate, best_T_candidate = best_R.clone(), best_T.clone() # Store candidates

    for _ in range(max_iter):
        indices = torch.randint(0, num_points, (N,), device=src.device)
        src_sample, target_sample = src[indices], target[indices]

        try:
            R_sample, T_sample = svd_2(src_sample, target_sample)
        except (torch.linalg.LinAlgError, AssertionError): # Catch SVD or assertion errors
            continue

        transformed_src = src @ R_sample.T + T_sample
        distances = torch.norm(transformed_src - target, dim=1)
        current_inliers_mask = distances < distance_threshold
        current_inlier_count = torch.sum(current_inliers_mask).item()

        if current_inlier_count > best_inlier_count:
            best_inlier_count = current_inlier_count
            best_R_candidate = R_sample
            best_T_candidate = T_sample
            best_inliers_mask_candidate = current_inliers_mask # Store mask for refitting

    if best_inlier_count >= min_inliers:
        try:
            best_R, best_T = svd_2(src[best_inliers_mask_candidate], target[best_inliers_mask_candidate])
            best_inliers_mask = best_inliers_mask_candidate
            print(f"RANSAC: Refit with {best_inlier_count}/{num_points} inliers.") 
        except (torch.linalg.LinAlgError, AssertionError):
            # print(f"Warning: RANSAC refitting failed. Using model from sample.") 
            best_R, best_T = best_R_candidate, best_T_candidate # Use sample model if refit fails
            best_inliers_mask = best_inliers_mask_candidate
    elif best_inlier_count >= 0: # Found some model, but less than min_inliers
         # print(f"Warning: RANSAC found only {best_inlier_count} inliers (< {min_inliers}). Using best sample model.") # Optional logging
         best_R, best_T = best_R_candidate, best_T_candidate
         best_inliers_mask = best_inliers_mask_candidate # Return the mask for the best sample found
    else: # No valid model found in any iteration
        # print(f"Warning: RANSAC failed to find any valid model. Using SVD on all points as fallback.") 
        try:
            best_R, best_T = svd_2(src, target)
            best_inliers_mask = torch.ones(num_points, dtype=torch.bool, device=src.device)
        except (torch.linalg.LinAlgError, AssertionError):
             # print(f"Error: SVD failed even with all points. Returning identity.") 
             best_R = torch.eye(3, device=src.device, dtype=src.dtype)
             best_T = torch.zeros(3, device=src.device, dtype=src.dtype)
             best_inliers_mask = torch.zeros(num_points, dtype=torch.bool, device=src.device)

    return best_R, best_T, best_inliers_mask



def chamfer_distance(pc1, pc2):
    # pc1, pc2: [B, N, 3], [B, M, 3]
    dist = torch.cdist(pc1, pc2)  # [B, N, M]
    min_dist_pc1_to_pc2 = torch.min(dist, dim=2)[0]  # [B, N]
    min_dist_pc2_to_pc1 = torch.min(dist, dim=1)[0]  # [B, M]
    return min_dist_pc1_to_pc2.mean() + min_dist_pc2_to_pc1.mean()

def multi_scale_cd(pc1, pc2, scales=[1.0, 0.5, 0.25]):
    loss = 0
    B, N, _ = pc1.shape
    for w in scales:
        M = int(N * w)
        idx1 = torch.randperm(N, device=pc1.device)[:M]
        idx2 = torch.randperm(N, device=pc2.device)[:M]
        loss += chamfer_distance(pc1[:, idx1], pc2[:, idx2])
    return loss / len(scales)

def approximate_emd(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 0.3,
    max_iter: int = 50,
    eps: float = 1e-9,
) -> torch.Tensor:

    B, N, _ = pred.shape
    M = target.shape[1]

    # 1) 计算平方距离矩阵 C
    C = torch.cdist(pred, target, p=2) ** 2  # (B, N, M)


    K = torch.exp(-C / lambda_reg)  # (B, N, M)

    # 3) 初始化 marginals a, b（均匀分布）
    a = torch.full((B, N), 1.0 / N, device=pred.device, dtype=pred.dtype)
    b = torch.full((B, M), 1.0 / M, device=pred.device, dtype=pred.dtype)

    # 4) Sinkhorn 向量 u, v 初始化
    u = torch.ones_like(a)
    v = torch.ones_like(b)


    for _ in range(max_iter):

        u = a / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + eps)

        v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + eps)

    T = u.unsqueeze(2) * K * v.unsqueeze(1)  # (B, N, M)

    emd = (T * C).sum(dim=(1, 2))  # (B,)

    # 返回批次均值
    return emd.mean()

def repulsion_loss(pc, k=5, h=0.01, eps=1e-12):
    """
    计算排斥损失，以鼓励点之间的均匀间距。

    参数:
        pc (torch.Tensor): 预测的点云 (B, N, 3)。
        k (int): 用于排斥的最近邻居数量。
        h (float): 排斥半径。距离小于 h 的点会受到惩罚。
        eps (float): 用于数值稳定性的小值。

    返回:
        torch.Tensor: 排斥损失值 (标量)。
    """
    B, N, _ = pc.shape
    if N <= k: # 处理点数少于 k 的情况
        return torch.tensor(0.0, device=pc.device, dtype=pc.dtype)

    dist_sq = torch.cdist(pc, pc)**2 # (B, N, N)，计算点对之间的平方距离
    # 查找每个点的 k+1 个最近邻居（包括自身）
    knn_dist_sq, _ = torch.topk(dist_sq, k + 1, dim=2, largest=False) # (B, N, k+1)
    # 排除自身距离 (索引为 0 的是自身，距离为 0)
    knn_dist_sq = knn_dist_sq[:, :, 1:] # (B, N, k)

    # 对半径 h 内的邻居进行惩罚
    # 使用 sqrt 计算距离，clamp 确保 sqrt 前距离非负
    knn_dist = torch.sqrt(torch.clamp(knn_dist_sq, min=0.0) + eps) # (B, N, k)
    # 惩罚 = max(0, h - distance)
    penalty = torch.clamp(h - knn_dist, min=0.0) # (B, N, k)

    # 对每个点的 k 个邻居的惩罚求和，然后在所有点和批次上取平均
    loss = torch.mean(penalty)
    return loss

def hausdorff_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    计算两个点云之间的 Hausdorff Distance。

    HD(A, B) = max(h(A, B), h(B, A))
    其中 h(A, B) = max_{a in A} { min_{b in B} { ||a - b||_2 } }

    参数:
        pc1 (torch.Tensor): 第一个点云 (B, N, 3)。
        pc2 (torch.Tensor): 第二个点云 (B, M, 3)。

    返回:
        torch.Tensor: 批次平均的 Hausdorff Distance (标量)。
    """
    B, N, _ = pc1.shape
    M = pc2.shape[1]

    # 计算成对距离矩阵 (使用 L2 范数)
    dist = torch.cdist(pc1, pc2, p=2)  # (B, N, M)

    # 计算 h(pc1, pc2)
    # 对 pc1 中的每个点，找到其到 pc2 中最近点的距离
    min_dist_pc1_to_pc2 = torch.min(dist, dim=2)[0]  # (B, N)
    # 找到 pc1 中离 pc2 最远点的距离
    h_pc1_to_pc2 = torch.max(min_dist_pc1_to_pc2, dim=1)[0]  # (B,)

    # 计算 h(pc2, pc1)
    # 对 pc2 中的每个点，找到其到 pc1 中最近点的距离
    min_dist_pc2_to_pc1 = torch.min(dist, dim=1)[0]  # (B, M)
    # 找到 pc2 中离 pc1 最远点的距离
    h_pc2_to_pc1 = torch.max(min_dist_pc2_to_pc1, dim=1)[0]  # (B,)

    # Hausdorff Distance 是两个方向上的最大值
    hd = torch.max(h_pc1_to_pc2, h_pc2_to_pc1)  # (B,)

    # 返回批次平均值
    return hd.mean()


def icp_refine(src: torch.Tensor, 
              target: torch.Tensor,
              max_iterations: int = 20,
              distance_threshold: float = 0.005,
              convergence_threshold: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用迭代最近点(ICP)算法细化源点云和目标点云之间的对齐。
    
    参数:
        src: 源点云，形状 (N, 3)
        target: 目标点云，形状 (M, 3)
        max_iterations: 最大迭代次数
        distance_threshold: 对应点考虑的最大距离阈值
        convergence_threshold: 收敛阈值，当变换增量小于此值时停止迭代
        
    返回:
        R: 旋转矩阵，形状 (3, 3)
        T: 平移向量，形状 (3,)
    """
    # 初始化变换
    R_current = torch.eye(3, device=src.device, dtype=src.dtype)
    T_current = torch.zeros(3, device=src.device, dtype=src.dtype)
    
    # 保存转换后的源点
    src_transformed = src.clone()
    
    # 迭代优化
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # 1. 为每个源点找到最近的目标点
        # 计算当前转换的源点和目标点之间的距离
        distances = torch.cdist(src_transformed, target)
        
        # 获取每个源点的最近目标点的索引
        min_distances, indices = torch.min(distances, dim=1)
        
        # 使用距离阈值筛选对应点
        valid_indices = min_distances < distance_threshold
        
        # 如果有效对应点太少，则提前退出
        if valid_indices.sum() < 3:
            print(f"ICP 迭代 {iteration}: 有效对应点不足 (只有 {valid_indices.sum()})，提前退出")
            break
            
        # 提取有效的源点和它们的目标对应点
        valid_src = src_transformed[valid_indices]
        valid_target = target[indices[valid_indices]]
        
        # 2. 估计最优变换
        try:
            R_step, T_step = svd_2(valid_src, valid_target)
        except Exception as e:
            print(f"ICP SVD 步骤失败: {e}")
            break
            
        # 3. 应用新的变换到源点
        src_transformed = src_transformed @ R_step.T + T_step
        
        # 4. 更新累积变换
        R_current = R_step @ R_current
        T_current = R_step @ T_current + T_step
        
        # 5. 检查收敛性（使用平均对应点距离）
        current_error = min_distances[valid_indices].mean().item()
        
        # 计算误差变化并检查收敛性
        error_change = abs(prev_error - current_error)
        
        if iteration % 5 == 0 or error_change < convergence_threshold:
            print(f"ICP 迭代 {iteration}: 平均误差 = {current_error:.6f}, 变化 = {error_change:.6f}")
            
        if error_change < convergence_threshold:
            print(f"ICP 已收敛，在迭代 {iteration}/{max_iterations} 后停止")
            break
            
        prev_error = current_error
    
    # 确保返回的旋转矩阵满足正交性和行列式为1
    # 通常SVD应该保证这一点，但为了稳健性我们再次检查
    U, S, V = torch.svd(R_current)
    R_current = U @ V.T
    
    # 处理镜像情况
    if torch.det(R_current) < 0:
        V_fixed = V.clone()
        V_fixed[:, 2] = -V_fixed[:, 2]
        R_current = U @ V_fixed.T
        
    return R_current, T_current


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.chamfer_alpha = 0.3
        self.pointMLP_dims = [3,64, 128, 1024]
        self.pointMLP = nn.ModuleList()

        for i in range(len(self.pointMLP_dims) - 1):
            self.pointMLP.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.pointMLP_dims[i], self.pointMLP_dims[i + 1], kernel_size=1
                    ),
                    nn.BatchNorm1d(self.pointMLP_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        self.coordMLP_dims = [64+1024,512,256,128,3]
        self.coordMLP = nn.ModuleList()
        for i in range(len(self.coordMLP_dims) - 1):
            if i == len(self.coordMLP_dims) - 2:
                self.coordMLP.append(
                    nn.Sequential(
                        nn.Conv1d(
                            self.coordMLP_dims[i], self.coordMLP_dims[i + 1], kernel_size=1
                        ),
                    )
                )
            else:
                self.coordMLP.append(
                    nn.Sequential(
                        nn.Conv1d(
                            self.coordMLP_dims[i], self.coordMLP_dims[i + 1], kernel_size=1
                        ),
                        nn.BatchNorm1d(self.coordMLP_dims[i + 1]),
                        nn.ReLU(),
                        # nn.Dropout(p=0.05)
                    )
                )


    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
    
        B, N, _ = pc.shape

        x = pc.permute(0, 2, 1) # (B,3,N) for conv
        local_features = None

        for layer in self.pointMLP:
            x = layer(x)
            if layer == self.pointMLP[0]:
                local_features = x # (B,64,N)

        global_feature = torch.max(x,dim=2)[0] # (B,1024)

        global_feature = global_feature.unsqueeze(-1).expand(-1,-1, N) # (B,1024,N)

        combined_features = torch.cat((local_features, global_feature), dim=1) # (B, 64+1024, N)
        x = combined_features
        for layer in self.coordMLP:
            x = layer(x)
        x = x.permute(0, 2, 1) # (B,N,3)

        pred_coords = x # (B,N,3)

        # if B > 0 and kwargs.get('visualize_forward', False) and False:
        #     # Use torch.no_grad and detach() to prevent gradient issues during visualization
        #     with torch.no_grad():
        #         print("Visualizing forward pass sample 0...")
        #         # Get first sample
        #         choice = random.randint(0, B)

        #         pc_0 = pc[choice]
        #         coord_0 = coord[choice]
        #         pred_0 = pred_coords[choice]

        #         # Convert to numpy
        #         pc_np = pc_0.detach().cpu().numpy()
        #         coord_np = coord_0.detach().cpu().numpy()
        #         pred_np = pred_0.detach().cpu().numpy()

        #         # Create visualization list
        #         vis_list = []
        #         vis_list.extend(Vis.pc(pc=pc_np, color='blue', size=2))      # Input Camera PC
        #         vis_list.extend(Vis.pc(pc=coord_np, color='yellow', size=2)) # Ground Truth Object Coords
        #         vis_list.extend(Vis.pc(pc=pred_np, color='red', size=2))      # Predicted Object Coords

        #         # Show
        #         Vis.show(vis_list)
        #         # Optional: Prevent showing again in the same run if called repeatedly
        #         # kwargs['visualize_forward'] = False

        
        chamfer_loss = chamfer_distance(pred_coords, coord)
        rep_loss = repulsion_loss(pred_coords)
        mse_loss = torch.nn.functional.mse_loss(pred_coords, coord) 
        hausdorff_loss = hausdorff_distance(pred_coords, coord)
        l1_loss = torch.nn.functional.smooth_l1_loss(pred_coords, coord)
        # emd_loss = approximate_emd(pred_coords, coord) 

        # loss = chamfer_loss  # (B,3) -> (B,1)
        # loss = mse_loss
        loss =  10 * torch.sqrt(mse_loss)
        # loss = multi_scale_cd(pred_coords, coord)
        # loss = approximate_emd(pred_coords, coord, lambda_reg=1.0, max_iter=50)
        metric = dict(
            loss=loss,
            # additional metrics you want to log
            chamfer_loss=chamfer_loss,
            # repulsion_loss=rep_loss,
            mse_loss = torch.sqrt( mse_loss),
            hausdorff_loss=hausdorff_loss,
            l1_loss = l1_loss,
            # emd_loss = emd_loss,
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        
        self.eval()
        with torch.no_grad():
            x = pc.permute(0, 2, 1) # (B,3,N) for conv
            local_features = None
            for layer in self.pointMLP:
                x = layer(x)
                if layer == self.pointMLP[0]:
                    local_features = x
            global_feature = torch.max(x,dim=2)[0]

            global_feature = global_feature.unsqueeze(-1).expand(-1,-1, pc.shape[1]) # (B,1024,N)
            combined_features = torch.cat((local_features, global_feature), dim=1)
            x = combined_features
            for layer in self.coordMLP:
                x = layer(x)
            x = x.permute(0, 2, 1) # (B,N,3)

            R_list = []
            T_list = []
            
            B = x.shape[0]
            for b in range(B):
                src_b = x[b]
                target_b = pc[b]
                # We need R,T from camera frame to object frame So
                # p^{camera} = R_{camera->object}^{camera} * p^{object} + T_{camera->object}^camera
                # We fit this R,T

                R_init, T_init = svd_2(src_b, target_b)

                # R_init,T_init,_ = ransac(src_b,target_b)

                R = R_init
                T = T_init
                               
                R_list.append(R)
                T_list.append(T)

                if b == 0 :
                    print("Visualizing sample 0...")
                    # Convert tensors to numpy arrays (move to CPU first)
                    pc_np = target_b.cpu().numpy()
                    pred_coords_np = src_b.cpu().numpy()
                    R_np = R.cpu().numpy()
                    T_np = T.cpu().numpy()

                    # Transform predicted coordinates back to camera frame using estimated R, T
                    # pc_aligned_np = (R_np @ pred_coords_np.T).T + T_np
                    # Correct matrix multiplication for (N, 3) points: R @ p^T requires p to be (3, N)
                    # Or directly: p @ R^T for p as (N, 3)
                    pc_aligned_np = pred_coords_np @ R_np.T + T_np


                    # Create visualization objects
                    vis_list = []
                    vis_list.extend(Vis.pc(pc=pc_np, color='blue', size=2)) # Input PC (target)
                    vis_list.extend(Vis.pc(pc=pred_coords_np, color='red', size=2)) # Predicted Coords (source)
                    vis_list.extend(Vis.pc(pc=pc_aligned_np, color='green', size=2)) # Predicted Coords aligned to Input PC
                    vis_list.extend(Vis.pose(trans=T_np, rot=R_np, length=0.1)) # Estimated Pose

                    # Show visualization
                    Vis.show(vis_list) # Opens in browser by default
                    # Or save to file: Vis.show(vis_list, path="output/vis_sample_0.html")    
            R = torch.stack(R_list)  # Stack the rotation matrices
            T = torch.stack(T_list)  # Stack the translation vectors
        return T, R
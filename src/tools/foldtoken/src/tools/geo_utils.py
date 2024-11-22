import torch
import numpy as np
from torch_scatter import scatter_sum, scatter_max, scatter_softmax, scatter_mean
from src.tools.affine_utils import quat_to_rot

def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

    
def angle(p, eps=1e-10):
    # p: [*, 3, 3]
    u1 = p[..., 0, :] - p[..., 1, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u1_norm = (eps + torch.sum(u1 ** 2, dim=-1)) ** 0.5
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5

    u1 = u1/u1_norm.unsqueeze(-1)
    u2 = u2/u2_norm.unsqueeze(-1)
    
    cos = torch.sum(u1*u2, dim=-1)
    return cos

def dihedral(p, eps=1e-10):
    # p: [*, 4, 3]

    # [*, 3]
    u1 = p[..., 1, :] - p[..., 0, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u3 = p[..., 3, :] - p[..., 2, :]

    # [*, 3]
    u1xu2 = torch.cross(u1, u2, dim=-1)
    u2xu3 = torch.cross(u2, u3, dim=-1)

    # [*]
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5
    u1xu2_norm = (eps + torch.sum(u1xu2 ** 2, dim=-1)) ** 0.5
    u2xu3_norm = (eps + torch.sum(u2xu3 ** 2, dim=-1)) ** 0.5

    # [*]
    cos_enc = torch.einsum('...d,...d->...', u1xu2, u2xu3)/ (u1xu2_norm * u2xu3_norm)
    sin_enc = torch.einsum('...d,...d->...', u2, torch.cross(u1xu2, u2xu3, dim=-1)) /  (u2_norm * u1xu2_norm * u2xu3_norm)

    return torch.stack([cos_enc, sin_enc], dim=-1)


def compute_frenet_frames(x, mask, eps=1e-10):
    # x: [b, n_res, 3]

    t = x[:, 1:] - x[:, :-1] # relative direction
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    n = torch.cross(t[:, :-1], t[:, 1:])
    n_norm = torch.sqrt(eps + torch.sum(n ** 2, dim=-1))
    n = n / n_norm.unsqueeze(-1)

    b = torch.cross(n, t[:, 1:])

    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # TODO: recheck correctness of this implementation
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]
        rots_[0] = rots_[1]
        rots_[length-1] = rots_[length-2]
        rots.append(rots_)
    rots = torch.stack(rots, dim=0).to(x.device)

    return rots

def kabsch_algorithm(P, Q):
    """
    使用 Kabsch 算法对齐两个点集 P 和 Q。
    """
    # 计算质心
    mu_P = P.mean(dim=0)
    mu_Q = Q.mean(dim=0)
    
    # 中心化点集
    P_centered = P - mu_P
    Q_centered = Q - mu_Q
    
    # 计算协方差矩阵
    H = torch.matmul(P_centered.t(), Q_centered)
    
    # SVD 分解
    U, S, Vt = torch.linalg.svd(H)
    
    # 计算旋转矩阵
    R = torch.matmul(Vt.t(), U.t())
    
    # 确保旋转矩阵是有效的
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(Vt.t(), U.t())
    
    # 计算平移向量
    t = mu_Q - torch.matmul(R, mu_P)
    
    return R, t

def kabsch_algorithm_batch(P, Q, batch_id):
    """
    使用 Kabsch 算法对齐两个点集 P 和 Q。
    """
    # torch.save({'P':P, 'Q':Q, 'batch_id':batch_id}, '/huyuqi/xmyu/FoldToken2/kabsch.pt')
    # 计算质心
    mu_P = scatter_mean(P, batch_id, dim=0)
    mu_Q = scatter_mean(Q, batch_id, dim=0)
    
    # 中心化点集
    P_centered = P - mu_P[batch_id]
    Q_centered = Q - mu_Q[batch_id]
    
    # 计算协方差矩阵
    # H = torch.matmul(P_centered.t(), Q_centered)
    H = scatter_sum(P_centered[:,:,None]*Q_centered[:,None], batch_id, dim=0)
    
    # SVD 分解
    U, S, Vt = torch.linalg.svd(H)
    
    # 计算旋转矩阵
    R = torch.matmul(Vt.permute(0,2,1), U.permute(0,2,1))
    
    # 确保旋转矩阵是有效的
    Vt[:, -1, :] *= torch.det(R)[:,None]
    R = torch.matmul(Vt.permute(0,2,1), U.permute(0,2,1))
    
    # 计算平移向量
    t = mu_Q - torch.matmul(R, mu_P[:,:,None])[:,:,0]
    
    return R, t

def batch_rmsd(P, Q, batch_id, MSE=False):
    """
    计算两个点集 P 和 Q 的 RMSD。
    """
    # 计算旋转矩阵和平移向量
    if len(P.shape)==3:
        with torch.no_grad():
            R, t = kabsch_algorithm_batch(P[:,1], Q[:,1], batch_id)

        # 对 P 应用变换
        P = torch.einsum('nki, nij->nkj', P, R.permute(0,2,1)[batch_id]) + t[batch_id][:,None]

        if MSE:
            return scatter_mean(torch.sum((P - Q)**2, dim=(1,2)), batch_id)
        else:
            return torch.sqrt(scatter_mean(torch.sum((P - Q)**2, dim=(1,2)), batch_id))
    
    if len(P.shape)==2:
        with torch.no_grad():
            R, t = kabsch_algorithm_batch(P, Q, batch_id)
        P = torch.einsum('ni, nij->nj', P, R.permute(0,2,1)[batch_id]) + t[batch_id]
        if MSE:
            return scatter_mean(torch.sum((P - Q)**2, dim=(1)), batch_id)
        else:
            return torch.sqrt(scatter_mean(torch.sum((P - Q)**2, dim=(1)), batch_id))
    


import torch
import torch.nn as nn
import numpy as np

class CrossRMSD(nn.Module):
    def __init__(self, method="power", method_iter=50):
        super(CrossRMSD, self).__init__()
        self.method = method
        self.method_iter = method_iter
        self._eps = 1e-5

        # R_to_F converts xyz cross-covariance matrices (3x3) to the (4x4) F
        # matrix of Coutsias et al. This F matrix encodes the optimal RMSD in
        # its spectra; namely, the eigenvector associated with the most
        # positive eigenvalue of F is the quaternion encoding the optimal
        # 3D rotation for superposition.
        # fmt: off
        R_to_F = np.zeros((9, 16)).astype("f")
        F_nonzero = [
        [(0,0,1.),(1,1,1.),(2,2,1.)],            [(1,2,1.),(2,1,-1.)],            [(2,0,1.),(0,2,-1.)],            [(0,1,1.),(1,0,-1.)],
                [(1,2,1.),(2,1,-1.)],  [(0,0,1.),(1,1,-1.),(2,2,-1.)],             [(0,1,1.),(1,0,1.)],             [(0,2,1.),(2,0,1.)],
                [(2,0,1.),(0,2,-1.)],             [(0,1,1.),(1,0,1.)],  [(0,0,-1.),(1,1,1.),(2,2,-1.)],             [(1,2,1.),(2,1,1.)],
                [(0,1,1.),(1,0,-1.)],             [(0,2,1.),(2,0,1.)],             [(1,2,1.),(2,1,1.)],  [(0,0,-1.),(1,1,-1.),(2,2,1.)]
        ]
        # fmt: on

        for F_ij, nonzero in enumerate(F_nonzero):
            for R_i, R_j, sign in nonzero:
                R_to_F[R_i * 3 + R_j, F_ij] = sign
        self.register_buffer("R_to_F", torch.tensor(R_to_F))

    def forward(self, X_mobile, X_target, batch_id):
        B = batch_id.shape[0]
        # Get unique batch IDs
        unique_batches = torch.unique(batch_id)

        num_source = len(unique_batches)

        # Center coordinates
        X_mobile_centered = X_mobile.clone()
        X_target_centered = X_target.clone()


        mu_mobile = scatter_mean(X_mobile, batch_id, dim=0)
        mu_target = scatter_mean(X_target, batch_id, dim=0)
    
        X_mobile_centered -= mu_mobile[batch_id]
        X_target_centered -= mu_target[batch_id]

        # CrossCov matrices contract over atoms
        R = scatter_sum(X_mobile_centered[:,:,None]*X_target_centered[:,None], batch_id, dim=0)
        R_flat = R.reshape(B, 9)
        F = torch.matmul(R_flat, self.R_to_F).reshape(B, 4, 4)

        # Compute optimal quaternion
        if self.method == "symeig":
            top_eig = torch.linalg.eigvalsh(F)[:, 3]
        elif self.method == "power":
            top_eig, vec = eig_leading(F, num_iterations=self.method_iter)
        else:
            raise NotImplementedError

        # Compute RMSD
        num_atoms = scatter_sum(torch.ones_like(batch_id), batch_id, dim=0)
        norms = (X_mobile ** 2).sum() + (X_target ** 2).sum()
        sqRMSD = torch.relu((norms - 2 * top_eig) / (num_atoms + self._eps))
        RMSD = torch.sqrt(sqRMSD)

        return RMSD

def eig_leading(F, num_iterations=50):
    batch_size = F.shape[0]
    vec = torch.randn(batch_size, 4, device=F.device)
    for _ in range(num_iterations):
        vec = torch.matmul(F, vec.unsqueeze(-1)).squeeze(-1)
        vec = torch.nn.functional.normalize(vec)
    top_eig = (vec.unsqueeze(1) @ F @ vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return top_eig, vec

if __name__ == '__main__':
    data = torch.load('/huyuqi/xmyu/FoldToken2/kabsch.pt')
    P, Q, batch_id = data['P'], data['Q'], data['batch_id']
    batch_rmsd(P,Q,batch_id)
    # R, t = kabsch_algorithm_batch(P, Q, batch_id)
    # R2, t2 = kabsch_algorithm(P[batch_id==0], Q[batch_id==0])
    print()
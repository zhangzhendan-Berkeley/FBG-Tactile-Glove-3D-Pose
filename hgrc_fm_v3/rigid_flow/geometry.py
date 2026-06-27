# rigid_flow/geometry.py
# -*- coding: utf-8 -*-
import torch, math

# ---------- basic utils ----------
def unit_vector(v: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Normalize vector along last dimension safely."""
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

# ---------- quaternion / rotation ----------
def quat_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Normalize quaternion (x,y,z,w)."""
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """q: (...,4) = (x,y,z,w) -> R: (...,3,3). Auto-normalizes q for robustness."""
    q = quat_normalize(q)
    x,y,z,w = q.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = torch.stack([
        1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy),
    ], dim=-1).reshape(q.shape[:-1]+(3,3))
    return R

def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (x,y,z,w). Normalizes output."""
    B = R.shape[:-2]
    q = torch.zeros(B+(4,), device=R.device, dtype=R.dtype)
    t = R[...,0,0] + R[...,1,1] + R[...,2,2]
    cond = t > 0
    if cond.any():
        t_pos = t[cond]
        r = torch.sqrt(1.0 + t_pos)
        w = 0.5 * r
        s = 0.5 / r
        q_pos = torch.stack([
            (R[...,2,1][cond] - R[...,1,2][cond]) * s,
            (R[...,0,2][cond] - R[...,2,0][cond]) * s,
            (R[...,1,0][cond] - R[...,0,1][cond]) * s,
            w
        ], dim=-1)
        q[cond] = q_pos
    if (~cond).any():
        Rc = R[~cond]
        i = torch.argmax(torch.stack([Rc[...,0,0], Rc[...,1,1], Rc[...,2,2]], dim=-1), dim=-1)
        x = torch.zeros((Rc.shape[0],4), device=R.device, dtype=R.dtype)
        for idx in range(3):
            mask = (i==idx)
            if not mask.any(): continue
            m = Rc[mask]
            if idx==0:
                r = torch.sqrt(1.0 + m[...,0,0] - m[...,1,1] - m[...,2,2])
                s = 0.5 / r
                x[mask] = torch.stack([
                    0.5*r,
                    (m[...,0,1]-m[...,1,0])*s,
                    (m[...,0,2]-m[...,2,0])*s,
                    (m[...,2,1]-m[...,1,2])*s
                ], dim=-1)
            elif idx==1:
                r = torch.sqrt(1.0 - m[...,0,0] + m[...,1,1] - m[...,2,2])
                s = 0.5 / r
                x[mask] = torch.stack([
                    (m[...,0,1]-m[...,1,0])*s,
                    0.5*r,
                    (m[...,1,2]-m[...,2,1])*s,
                    (m[...,0,2]-m[...,2,0])*s
                ], dim=-1)
            else:
                r = torch.sqrt(1.0 - m[...,0,0] - m[...,1,1] + m[...,2,2])
                s = 0.5 / r
                x[mask] = torch.stack([
                    (m[...,0,2]-m[...,2,0])*s,
                    (m[...,1,2]-m[...,2,1])*s,
                    0.5*r,
                    (m[...,1,0]-m[...,0,1])*s
                ], dim=-1)
        q[~cond] = x
    return quat_normalize(q)

def rot_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D rotation representation: use first two columns."""
    c1 = R[...,0]; c2 = R[...,1]
    return torch.cat([c1, c2], dim=-1)

def r6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Inverse of 6D representation (stable Gram-Schmidt)."""
    a1 = d6[...,0:3]; a2 = d6[...,3:6]
    b1 = unit_vector(a1)
    b2 = unit_vector(a2 - (b1*a2).sum(-1,keepdim=True)*b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1,b2,b3], dim=-1)

# ---------- coordinate remap: xyz -> yzx ----------
# 核心：位置 p' = P p；旋转 R' = P R P^T；6D/四元数都必须经由“矩阵中转”

def _perm_mat_yzx(device=None, dtype=None):
    """Permutation matrix for xyz -> yzx."""
    return torch.tensor([[0,1,0],
                         [0,0,1],
                         [1,0,0]], device=device, dtype=dtype)

def vec_xyz_to_yzx(p: torch.Tensor) -> torch.Tensor:
    """位置坐标重映射：xyz -> yzx"""
    P = _perm_mat_yzx(device=p.device, dtype=p.dtype)
    return torch.einsum('ij,...j->...i', P, p)

def rot_xyz_to_yzx(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵重映射：R' = P R P^T，xyz -> yzx"""
    P = _perm_mat_yzx(device=R.device, dtype=R.dtype)
    return torch.einsum('ia,...ab,bj->...ij', P, R, P.t())

def quat_xyz_to_yzx(q: torch.Tensor) -> torch.Tensor:
    """四元数重映射（经由矩阵中转），输入输出均为 (x,y,z,w)"""
    R = quat_to_matrix(q)
    R2 = rot_xyz_to_yzx(R)
    return matrix_to_quat(R2)

def r6d_xyz_to_yzx(d6: torch.Tensor) -> torch.Tensor:
    """6D旋转重映射（经由矩阵中转），xyz -> yzx"""
    R = r6d_to_matrix(d6)
    R2 = rot_xyz_to_yzx(R)
    return rot_to_6d(R2)

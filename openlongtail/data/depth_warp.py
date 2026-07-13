from __future__ import annotations
import torch

def se3_inverse_torch(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Tinv = torch.zeros_like(T)
    Tinv[..., :3, :3] = R.transpose(-1, -2)
    Tinv[..., :3, 3] = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    Tinv[..., 3, 3] = 1.0
    return Tinv

def forward_splat_warp(front_rgb: torch.Tensor, front_depth: torch.Tensor, K_front: torch.Tensor, K_target: torch.Tensor, T_front_to_target: torch.Tensor, out_h: int, out_w: int, splat_radius: int=1, near: float=0.5, far: float=200.0) -> tuple[torch.Tensor, torch.Tensor]:
    if front_rgb.dim() != 4 or front_rgb.shape[1] != 3:
        raise ValueError(f'expected front_rgb (T,3,H,W), got {tuple(front_rgb.shape)}')
    if front_depth.dim() != 3:
        raise ValueError(f'expected front_depth (T,H,W), got {tuple(front_depth.shape)}')
    if front_rgb.shape[0] != front_depth.shape[0]:
        raise ValueError('frame counts must match between rgb and depth')
    if front_rgb.shape[2:] != front_depth.shape[1:]:
        raise ValueError(f'spatial sizes must match: rgb {front_rgb.shape[2:]} vs depth {front_depth.shape[1:]}')
    device = front_rgb.device
    (T, _, H, W) = front_rgb.shape
    (fx, fy) = (float(K_front[0, 0]), float(K_front[1, 1]))
    (cx, cy) = (float(K_front[0, 2]), float(K_front[1, 2]))
    (fxt, fyt) = (float(K_target[0, 0]), float(K_target[1, 1]))
    (cxt, cyt) = (float(K_target[0, 2]), float(K_target[1, 2]))
    (vs, us) = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    us_flat = us.reshape(-1)
    vs_flat = vs.reshape(-1)
    rgb_flat = front_rgb.float().permute(0, 2, 3, 1).reshape(T, H * W, 3)
    d_flat = front_depth.reshape(T, H * W)
    Xn = (us_flat - cx) / fx
    Yn = (vs_flat - cy) / fy
    P_x = Xn.unsqueeze(0) * d_flat
    P_y = Yn.unsqueeze(0) * d_flat
    P_z = d_flat
    ones = torch.ones_like(P_z)
    P = torch.stack([P_x, P_y, P_z, ones], dim=-1)
    T_rel = T_front_to_target.to(device=device, dtype=torch.float32)
    Pt = torch.einsum('ij,thj->thi', T_rel, P)
    (Xt, Yt, Zt) = (Pt[..., 0], Pt[..., 1], Pt[..., 2])
    safe_Z = Zt.clamp(min=1e-06)
    ut_f = fxt * Xt / safe_Z + cxt
    vt_f = fyt * Yt / safe_Z + cyt
    ut = ut_f.round().long()
    vt = vt_f.round().long()
    valid = (Zt > near) & (d_flat > near) & (d_flat < far)
    in_bounds = (ut >= 0) & (ut < out_w) & (vt >= 0) & (vt < out_h)
    keep = valid & in_bounds
    warped = torch.zeros((T, out_h, out_w, 3), dtype=torch.float32, device=device)
    visibility = torch.zeros((T, out_h, out_w), dtype=torch.float32, device=device)
    z_buffer = torch.full((T, out_h, out_w), float('inf'), dtype=torch.float32, device=device)
    splat_offsets = [(du, dv) for du in range(-splat_radius, splat_radius + 1) for dv in range(-splat_radius, splat_radius + 1)]
    for t_idx in range(T):
        kt = keep[t_idx]
        if not kt.any():
            continue
        idxs = kt.nonzero(as_tuple=False).squeeze(-1)
        u_src = ut[t_idx][idxs]
        v_src = vt[t_idx][idxs]
        z_src = Zt[t_idx][idxs]
        rgb_src = rgb_flat[t_idx][idxs]
        order = torch.argsort(z_src, descending=True)
        u_sort = u_src[order]
        v_sort = v_src[order]
        z_sort = z_src[order]
        rgb_sort = rgb_src[order]
        for (du, dv) in splat_offsets:
            u_off = (u_sort + du).clamp(0, out_w - 1)
            v_off = (v_sort + dv).clamp(0, out_h - 1)
            valid_off = (u_sort + du >= 0) & (u_sort + du < out_w) & (v_sort + dv >= 0) & (v_sort + dv < out_h)
            u_off = u_off[valid_off]
            v_off = v_off[valid_off]
            z_off = z_sort[valid_off]
            rgb_off = rgb_sort[valid_off]
            warped[t_idx].index_put_((v_off, u_off, torch.zeros_like(u_off)), rgb_off[:, 0], accumulate=False)
            warped[t_idx].index_put_((v_off, u_off, torch.ones_like(u_off)), rgb_off[:, 1], accumulate=False)
            warped[t_idx].index_put_((v_off, u_off, torch.full_like(u_off, 2)), rgb_off[:, 2], accumulate=False)
            z_buffer[t_idx].index_put_((v_off, u_off), z_off, accumulate=False)
            visibility[t_idx].index_put_((v_off, u_off), torch.ones_like(z_off), accumulate=False)
    warped = warped.permute(0, 3, 1, 2).clamp(0, 255).to(torch.uint8)
    visibility = visibility.unsqueeze(1)
    return (warped, visibility)

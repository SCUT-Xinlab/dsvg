# ==========================================
# 1. 标准库导入
# ==========================================
import os
import logging
from argparse import Namespace
from typing import List, Tuple, Optional

# ==========================================
# 2. 第三方库导入
# ==========================================
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from tqdm import tqdm

# ==========================================
# 3. 内部与本地包导入
# ==========================================
from nesvor import svr
from nesvor.image import Volume, Slice, load_volume
from nesvor.inr import models, data
from nesvor.transform import axisangle2mat, RigidTransform
from nesvor.slice_acquisition import slice_acquisition, slice_acquisition_adjoint
from nesvor.utils import get_PSF
from nesvor.cli.commands import _register, _sample_inr

# 安全导入高阶训练模块
try:
    from nesvor.inr.train import fide_coarse_train, fide_refine_train
except ImportError as e:
    logging.warning(f"无法导入 nesvor.inr.train (功能受限): {e}")

from .trainer import GaussianDiffusion, default


# ==========================================
# 工具函数 (Utility Functions)
# ==========================================
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x.flatten(), y.flatten())

def CG(A, b, x0, n_iter):
    """共轭梯度法 (Conjugate Gradient) 求解"""
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new

def PSFreconstruction(transforms, slices, slices_mask, vol_mask, params):
    return slice_acquisition_adjoint(
        transforms, params['psf'], slices, slices_mask, vol_mask, 
        params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], True
    )

def crop_nonzero_tensor(tensor: torch.Tensor):
    """根据非零元素裁剪张量"""
    non_zero_indices = torch.nonzero(tensor)
    min_indices = non_zero_indices.min(dim=0).values
    max_indices = non_zero_indices.max(dim=0).values
    
    slices = [slice(min_indices[i].item(), max_indices[i].item() + 1) for i in range(tensor.dim())]
    cropped_tensor = tensor[slices]
    
    return cropped_tensor, non_zero_indices

def resize_target(img: torch.Tensor, prefix_shape: tuple, nz: torch.Tensor) -> torch.Tensor:
    min_indices = nz.min(dim=0).values
    max_indices = nz.max(dim=0).values
    cropped_shape = [max_indices[i].item() + 1 - min_indices[i].item() for i in range(1, len(min_indices))]
    
    img_croped = F.interpolate(img, size=cropped_shape, mode='trilinear', align_corners=False)
    img_ori = torch.zeros(prefix_shape, device=img.device)
    
    img_ori[
        min_indices[0].item():max_indices[0].item()+1,
        min_indices[1].item():max_indices[1].item()+1,
        min_indices[2].item():max_indices[2].item()+1,
        min_indices[3].item():max_indices[3].item()+1 
    ] = img_croped
    
    return img_ori

def resize_img_mask(img: torch.Tensor, batch_size: int, shape: tuple) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(0)
        prefix_shape = img.shape
    elif img.dim() == 4:
        if img.shape[-1] == 2 or img.shape[-1] == 1:
            img = img.permute(3, 0, 1, 2)
            if img.shape[0] == 2:
                img = img[0][None, ...]
        prefix_shape = img.shape
    elif img.dim() == 5:
        img = img.squeeze(0)
        prefix_shape = img.shape
    else:
        raise ValueError(f"不支持的张量形状 {img.shape}，期望 3D 或 4D 张量。")
        
    device = img.device
    img = img.cpu()
    img, nz = crop_nonzero_tensor(img)
    
    scalar_image = tio.ScalarImage(tensor=img)
    resize_transform = tio.Resize(tuple(shape))
    resized_image = resize_transform(scalar_image)
    resized_tensor = resized_image.tensor.to(device)

    if resized_tensor.dim() == 3:
        resized_tensor = resized_tensor.unsqueeze(0)
        
    resized_tensor = resized_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    return resized_tensor, prefix_shape, nz

def standardize_and_contrast_stretch(image: torch.Tensor) -> torch.Tensor:
    """标准化并拉伸对比度到 [0, 1]"""
    mean = image.mean()
    std = image.std()
    standardized_image = (image - mean) / (std + 1e-8)
    
    min_val = standardized_image.min()
    max_val = standardized_image.max()
    contrast_stretched_image = (standardized_image - min_val) / (max_val - min_val + 1e-8)
    return contrast_stretched_image

def normalize_and_standardize(image: torch.Tensor) -> torch.Tensor:
    """归一化到 [-1, 1] 并进行均值方差标准化"""
    min_val = image.min()
    max_val = image.max()
    normalized_image = 2 * (image - min_val) / (max_val - min_val + 1e-8) - 1
    
    mean = normalized_image.mean()
    std = normalized_image.std()
    standardized_image = (normalized_image - mean) / (std + 1e-8)
    return standardized_image

def save_path(img: torch.Tensor, path: str):
    """保存张量为 NIfTI 图像"""
    nifti_img = nib.Nifti1Image(img.squeeze().detach().cpu().numpy(), affine=np.eye(4))
    nib.save(nifti_img, path)


# ==========================================
# 核心网络/流水线模块
# ==========================================
class SRR(nn.Module):
    """超分辨率重建模块 (Super-Resolution Reconstruction)"""
    def __init__(self, n_iter=10, use_CG=False, alpha=0.5, beta=0.02, delta=0.1):
        super().__init__()
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta * delta * delta
        self.delta = delta
        self.use_CG = use_CG

    def forward(self, theta, slices, volume, params, p=None, mu=0, z=None, vol_mask=None, slices_mask=None):
        if len(theta.shape) == 2:
            transforms = axisangle2mat(theta)
        else:
            transforms = theta

        A = lambda x: self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x: self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x: self.AtA(transforms, x, vol_mask, slices_mask, p, params, mu, z)

        x = volume
        y = slices
        
        if self.use_CG:
            b = At(y * p if p is not None else y)
            if mu and z is not None:
                b = b + mu * z
            x = CG(AtA, b, volume, self.n_iter) 
        else:
            for _ in range(self.n_iter):
                err = A(x) - y
                if p is not None:
                    err = p * err
                g = At(err)
                if self.beta:
                    dR = self.dR(x, self.delta)
                    g.add_(dR, alpha=self.beta)
                x.add_(g, alpha=-self.alpha)
        return F.relu(x, inplace=True)
    
    def A(self, transforms, x, vol_mask, slices_mask, params):
        return slice_acquisition(transforms, x, vol_mask, slices_mask, params['psf'], params['slice_shape'], params['res_s'] / params['res_r'], False, params['interp_psf'])

    def At(self, transforms, x, slices_mask, vol_mask, params):
        return slice_acquisition_adjoint(transforms, params['psf'], x, slices_mask, vol_mask, params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], False)

    def AtA(self, transforms, x, vol_mask, slices_mask, p, params, mu, z):
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol

    def dR(self, v, delta):
        g = torch.zeros_like(v)
        D, H, W = v.shape[-3:]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    v0 = v[:, :, 1:D-1, 1:H-1, 1:W-1]
                    v1 = v[:, :, 1+dz:D-1+dz, 1+dy:H-1+dy, 1+dx:W-1+dx]
                    dv = v0 - v1
                    dv_ = dv * (1 / (dx*dx + dy*dy + dz*dz) / (delta*delta))
                    g[:, :, 1:D-1, 1:H-1, 1:W-1] += dv_ / torch.sqrt(1 + dv * dv_)
        return g


class FidePipe(object):
    """基于先验的体数据重建流水线"""
    def __init__(self, args: Namespace, bounding_box: torch.Tensor=None, spatial_scaling: float = 1.0, mode='svr') -> None:
        self.args = args
        self.mode = mode
        self._n_train = None
        self.srr = SRR(2)
        
    @property
    def n_train(self):
        return self._n_train
        
    @n_train.setter
    def n_train(self, ntrain):
        self._n_train = ntrain
        
    def __call__(self, x: Volume, slices: List[Slice], mask=None, USE_age=0) -> torch.Tensor:
        """
        处理主入口:
        x: 从扩散模型生成的 volume
        slices: 获取到的切片列表
        返回: fidelity volume
        """
        # 预处理：基于切片均值进行强度缩放
        sum_val = torch.tensor(0.0, device=slices[0].image.device, dtype=torch.float32)
        
        logging.debug(f"Slice max value: {slices[0].image.max().item():.4f}")
        for s in slices:
            sum_val += s.v_masked.mean()
            
        slice_mean = sum_val / len(slices)
        x.rescale(intensity_mean=slice_mean.cpu())
        
        # 路由到不同的重建算法
        if self.mode == 'nesvor':
            model = self.init_volume(x, self.args)
            output_volume = self.refine_volume_nesvor(model, slices, mask=mask, USE_age=USE_age)
            return output_volume
            
        elif self.mode == 'svr':
            output_volume = self.refine_volume_svr(x, slices)
            return output_volume
            
        elif self.mode == 'cg':
            theta = RigidTransform.cat([s.transformation for s in slices])
            res_s = slices[0].resolution_x
            res_r = getattr(self.args, 'output_resolution', 0.5)
            s_thick = slices[0].resolution_z
            psf = get_PSF(
                res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
                device=x.device,
            )
            params = {
                "psf": psf,
                "slice_shape": slices[0].img.shape,
                "interp_psf": False,
                "res_s": res_s,
                "res_r": res_r,
                "s_thick": s_thick,
                "volume_shape": x.shape,
            }
            output_volume = self.srr(theta, slices, x, params)
            return output_volume
        else:
            raise NotImplementedError(f"未知的运行模式: {self.mode}")

    def init_volume(self, x: torch.Tensor, args: Namespace) -> torch.nn.Module:
        model = fide_coarse_train(x, args, ntrain=self._n_train // 2)
        return model

    def refine_volume_nesvor(self, model: models.NeSVoR, slices: List[Slice], mask, USE_age) -> torch.Tensor:
        mask_atlas = mask if USE_age else None
        inr, slices_transform, xyz_mask = fide_refine_train(
            slices=slices, args=self.args, model_trained=model, ntrain=self._n_train, mask_atlas=mask_atlas
        )
        if not USE_age:
            xyz_mask = mask
            
        output_volume, simulated_slices = _sample_inr(
            self.args,
            inr,
            xyz_mask,
            slices_transform,
            getattr(self.args, "output_volume", None) is not None,
            getattr(self.args, "simulated_slices", None) is not None,
        )
        return output_volume

    def refine_volume_svr(self, x: torch.Tensor, slices: List[Slice]) -> torch.Tensor:
        output_volume, output_slices, simulated_slices = svr.slice_to_volume_reconstruction_with_atlas(
            x, slices=slices, **vars(self.args)
        )
        return output_volume


class FideGaussianDiffusion(GaussianDiffusion):
    """支持条件注入和流形投影的增强扩散模型"""
    
    @property
    def prefix_shape_nz_index(self):
        return self._prefix_shape_nz_index
        
    @prefix_shape_nz_index.setter
    def prefix_shape_nz_index(self, prefix_shape_nz):
        self._prefix_shape_nz_index = prefix_shape_nz
        
    @property
    def slices(self):
        return self._slices
        
    @slices.setter
    def slices(self, slices):
        self._slices = slices
        
    @property
    def svr_args(self):
        return self._svr_args
        
    @svr_args.setter
    def svr_args(self, args):
        self._svr_args = args
        
    @property
    def condition(self):
        return self._condition_mask
        
    @condition.setter
    def condition_transformation(self, mask):
        self._condition_mask = mask

    # ==========================================
    # 数据格式转换与处理辅助方法
    # ==========================================
    def _tensor_to_volume(self, img_tensor: torch.Tensor, condition_tensors: torch.Tensor) -> Volume:
        """内部方法: 将网络输出的连续张量 [-1, 1] 逆向转换为物理空间的 NIfTI Volume"""
        device = self.betas.device
        
        # 归一化反转: [-1, 1] -> [0, 1]
        img_unnorm = (img_tensor + 1) / 2
        img_resized = resize_target(img_unnorm, *self.prefix_shape_nz_index).to(device)
        
        # 掩码张量反转与缩放
        cond_mask_unnorm = (condition_tensors[:, :1, ...] + 1) / 2
        cond_mask_target = resize_target(cond_mask_unnorm, *self.prefix_shape_nz_index).squeeze()
        
        # 构建物理体数据对象
        diff_volume = Volume(
            image=img_resized.squeeze(), 
            mask=cond_mask_target.bool().to(device),
            transformation=self.condition.transformation, 
            resolution_x=0.5, resolution_y=0.5, resolution_z=0.5
        )
        return diff_volume

    def _volume_to_tensor(self, vol: Volume, batch_size: int) -> torch.Tensor:
        """内部方法: 将物理体数据重采样，并转换回网络所需的扩散张量 [-1, 1]"""
        img_post, _, _ = resize_img_mask(
            vol.image, 
            batch_size, 
            (self.image_size, self.image_size, self.depth_size)
        )
        
        # 归一化: [0, 1] -> [-1, 1]
        img_post = img_post * 2 - 1
        return img_post

    # ==========================================
    # 核心采样逻辑
    # ==========================================
    def fide_sample_process(self, shape, img, i, condition_tensors=None):
        """流形投影与 Fidelity 精化 (Manifold Projection & Fidelity Refinement)"""
        device = self.betas.device
        batch_size = img.shape[0]
        result_folder = os.path.dirname(self.svr_args.output_volume)
        save_intermediates = True  # 显式控制调试文件保存的开关

        # 1. 图像预处理 (Preprocessing: Tensor -> Volume)
        diff_volume = self._tensor_to_volume(img, condition_tensors)
        
        if i == 0:               
            return diff_volume
            
        if save_intermediates:
            diff_volume.save(os.path.join(result_folder, f'resample-diff_process_{i}.nii.gz'))

        # 2. 核心精化步骤 (Core Step: Fidelity Refinement)
        mode = 'nesvor' if getattr(self.svr_args, 'ablation', '') != 'ablation_mibr' else 'svr'
        fidepipe = FidePipe(self.svr_args, mode=mode)
        fidepipe.n_train = 2000
        
        output_volume = fidepipe(
            x=diff_volume, 
            slices=self.slices, 
            mask=self.condition, 
            USE_age=self.svr_args.use_age
        )
        
        if save_intermediates:
            output_volume.save(os.path.join(result_folder, f'resample-fide_{i}.nii.gz'))

        # 3. 图像后处理 (Postprocessing: Volume -> Tensor)
        img_post = self._volume_to_tensor(output_volume, batch_size)
        
        save_path(img_post, os.path.join(result_folder, f'resample-vt_{i}.nii.gz'))
        
        ablation = getattr(self.svr_args, 'ablation', '')
        if ablation in ['ablation_addnoise', 'ablation_mibr']:
            return img_post

        # 4. 加噪投影 (Add Noise / Q-Sample)
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        noise = default(None, lambda: torch.randn_like(img_post))
        
        img_noisy = self.q_sample(x_start=img_post, t=t_tensor, noise=noise)
        
        if save_intermediates:
            refined_volume = Volume(
                image=img_noisy.squeeze(),
                transformation=self.condition.transformation, 
                resolution_x=0.5, resolution_y=0.5, resolution_z=0.5
            )
            refined_volume.save(os.path.join(result_folder, f'resample-diff_refine_{i}.nii.gz'))
            
        return img_noisy

    def p_sample_loop(self, shape, condition_tensors=None):  
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        # 动态获取 pipeline 的触发间隔
        gap = getattr(self.svr_args, 'gap', 90)
        ablation = getattr(self.svr_args, 'ablation', '')
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling loop', total=self.num_timesteps):
            if self.with_condition:        
                with torch.no_grad():
                    t = torch.full((b,), i, device=device, dtype=torch.long)
                    img = self.p_sample(img, t, condition_tensors=condition_tensors)
                    
                # 注入先验 / Fidelity 步骤
                if i % gap == 0 and ablation != 'ablation_nesvor':
                    img = self.fide_sample_process(shape, img, i, condition_tensors=condition_tensors)
            else:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
                
        return img
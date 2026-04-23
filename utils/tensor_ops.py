# ==========================================
# 标准库与第三方库导入 (Standard & Third-party Imports)
# ==========================================
import numpy as np
import nibabel as nib
import torch
import torchio as tio
from torchvision.transforms import Compose, Lambda

# ==========================================
# 全局数据增强/变换定义 (Global Transforms)
# ==========================================
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.squeeze(0)),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.squeeze(0) if t.ndim == 5 else t),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
])

# ==========================================
# 张量与图像处理工具函数 (Tensor & Image Ops Utils)
# ==========================================
def save_tensor_as_nii(tensor, filename, affine=None):
    """
    将三维张量保存为 NIfTI 文件 (.nii 或 .nii.gz)。
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()  
    
    if tensor.ndim != 3:
        raise ValueError("输入张量必须是三维的，形状应为 (D, H, W)") 
    
    if affine is None:
        affine = np.eye(4)
        
    # 创建 NIfTI 图像对象并保存
    nifti_image = nib.Nifti1Image(tensor, affine)
    nib.save(nifti_image, filename)
    print(f"文件已保存为: {filename}")

def crop_nonzero_tensor(tensor):
    """裁剪张量，去除全为 0 的边界区域"""
    non_zero_indices = torch.nonzero(tensor)
    
    min_indices = non_zero_indices.min(dim=0).values
    max_indices = non_zero_indices.max(dim=0).values
    
    slices = [slice(min_indices[i].item(), max_indices[i].item() + 1) for i in range(tensor.dim())]
    cropped_tensor = tensor[slices]
    
    return cropped_tensor, non_zero_indices

def label2masks(masked_img, input_channel, batch_size=1):
    """将标签转换为 One-hot 形式的 Mask (目前逻辑针对特定场景)"""
    result_img = torch.ones(masked_img.shape + (input_channel - 1,)).to(masked_img.device)
    result_img[..., 0][masked_img == 0] = 0
    return result_img

def resize_img_mask(img: torch.Tensor, batch_size, shape) -> torch.Tensor:
    """使用 torchio 调整图像/Mask 尺寸，并处理不同维度的张量输入"""
    mask_ones = None
    if img.dim() == 3:
        img = img.unsqueeze(0)  # 从 (H, W, D) -> (C, H, W, D)
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
        raise ValueError(f"不支持的张量形状 {img.shape}，期望3D或4D张量。")
    
    device = img.device
    img = img.cpu()
    img, nz = crop_nonzero_tensor(img)
    scalar_image = tio.ScalarImage(tensor=img)
    
    resize_transform = tio.Resize(tuple(shape))
    resized_image = resize_transform(scalar_image)
    resized_tensor = resized_image.tensor.to(device)

    if resized_tensor.dim() == 3:
        resized_tensor = resized_tensor.unsqueeze(0)
        
    if mask_ones:
        resized_tensor = torch.cat([resized_tensor, -1 * torch.ones_like(resized_tensor)], dim=0)
        
    return resized_tensor, prefix_shape, nz
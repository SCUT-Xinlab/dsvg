# ==========================================
# 1. 标准库导入 (Standard Library Imports)
# ==========================================
import os
import sys
import argparse
import json 
import logging
from argparse import Namespace
from typing import List, Tuple

# ==========================================
# 2. 第三方库导入 (Third-Party Imports)
# ==========================================
import torch
import torchio as tio
from monai.utils import optional_import

# 初始化 tqdm
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

# ==========================================
# 3. 本地与项目包导入 (Local/Project Imports)
# ==========================================
from nesvor import svr
from nesvor.image import Slice, Volume, load_volume
from nesvor.inr import models
from nesvor.cli.commands import _register, _segment_stack
from nesvor.cli.parsers import main_parser
from nesvor.cli.io import inputs
from nesvor.cli.main import run

from diffusion_model.trainer_fide import FideGaussianDiffusion, resize_target
from diffusion_model.unet import create_model
from utils.script_process import Subject, DataSet, datesetloader

# 引入刚刚抽离的张量处理工具
from utils.tensor_ops import (
    transform, 
    input_transform, 
    save_tensor_as_nii, 
    crop_nonzero_tensor, 
    label2masks, 
    resize_img_mask
)
import argparse  # 确保顶部导入了 argparse

# ==========================================
# 命令行参数配置 (CLI Arguments)
# ==========================================
# ==========================================
# 命令行与 JSON 配置文件解析 (CLI & Config Parsing)
# ==========================================

def parse_pipeline_args():
    parser = argparse.ArgumentParser(description="Med-DDPM 胎儿大脑重建流水线")
    
    # 配置文件入口
    parser.add_argument("--config", type=str, default='/home/lvyao/git/med-ddpm/config/run_dataset_config/shegnfsy_0421.json', help="JSON 配置文件路径 (注: JSON配置的优先级高于命令行参数)")
    
    # IO 路径参数 (移除 required=True，因为它们可以从 JSON 读取)
    parser.add_argument("--input_dir", type=str, default=None, help="输入数据栈的文件夹路径 (sim_in)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出结果的文件夹路径 (sim_out)")
    parser.add_argument("--mask_dir", type=str, default=None, help="掩码数据的文件夹路径 (mask_folder)")
    parser.add_argument("--weight_path", type=str, default=None, help="扩散模型的权重文件路径 (.pt)")
    
    # 核心超参数 (保留默认值，作为 JSON 未配置时的兜底)
    parser.add_argument("--device", type=int, default=0, help="运行所使用的 GPU ID (默认: 0)")
    parser.add_argument("--timesteps", type=int, default=250, help="扩散模型的采样步数 (默认: 250)")
    parser.add_argument("--age", type=int, default=1, help="胎儿孕周 (0表示从数据集自动推断或禁用)")
    parser.add_argument("--dilate", type=int, default=3, help="Mask膨胀的迭代次数 (默认: 3)")
    parser.add_argument("--gap", type=int, default=90, help="Gap参数 (默认: 90)")
    
    # 运行模式与开关
    parser.add_argument("--mode", type=str, default="fide", choices=["fide", "nesvor", "test"], help="流水线运行模式")
    parser.add_argument("--fbs_seg", action="store_true", help="是否启用 FBS Segmentation")

    # 1. 优先解析命令行参数
    args = parser.parse_args()
    
    # 2. 如果提供了 JSON，读取并覆盖 (实现 JSON 优先级 > 命令行)
    if args.config:
        if not os.path.exists(args.config):
            parser.error(f"找不到配置文件: {args.config}")
            
        with open(args.config, 'r', encoding='utf-8') as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError as e:
                parser.error(f"解析 JSON 配置文件失败: {e}")
                
        for key, value in config_data.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"[Warning] JSON 配置文件中包含未知参数 '{key}'，已忽略。")

    # 3. 最终合法性校验 (检查必填路径是否至少在一个地方被赋值)
    missing_args = []
    if not args.input_dir: missing_args.append("input_dir")
    if not args.output_dir: missing_args.append("output_dir")
    if not args.weight_path: missing_args.append("weight_path")
    
    if missing_args:
        parser.error(f"缺少必填参数，请通过命令行或 JSON 配置文件提供: {', '.join(missing_args)}")

    return args

# ==========================================
# 核心业务类：dpmRecon
# ==========================================

class dpmRecon(object):
    def __init__(
        self, args: Namespace, input_size, num_channels, num_res_blocks, 
        num_class_labels, out_channels, depth_size, weightfile, mode='svr',
    ) -> None:
        self.args = args
        self.mode = mode
        self.device = args.device
        self.batch_size = 1
        self.channels = 1
        self.depth_size = depth_size
        self.image_size = input_size

        logging.info(f"初始化 U-Net 模型 (input_size={input_size}, depth_size={depth_size})...")
        model = create_model(input_size, num_channels, num_res_blocks, in_channels=num_class_labels, out_channels=out_channels).to(self.device)
        
        self.diffusion = FideGaussianDiffusion(
            model,
            image_size = input_size,
            depth_size = depth_size,
            timesteps = args.timesteps,
            loss_type = 'L1', 
            with_condition=True,
        ).to(self.device)
        
        logging.info(f"正在从 {weightfile} 加载模型权重...")
        # 增加 map_location 确保权重加载到正确的 device 上
        self.diffusion.load_state_dict(torch.load(weightfile, map_location=f'cuda:{self.device}')['ema'])
        logging.info("扩散模型加载成功！")

    def _get_atlas_path(self) -> str:
        """获取 Atlas 图谱路径"""
        base_dir = '/home/lvyao/local/atlas/CRL_FetalBrainAtlas_2017v3'
        if self.args.use_age:
            agestr = str(self.args.age) + 'exp' if self.args.age >= 36 else str(self.args.age)
            return f"{base_dir}/STA{agestr}.nii.gz"
        else:
            return f"{base_dir}/STA35.nii.gz"

    def _prepare_condition(self, slices: List[Slice]) -> Tuple[Volume, str]:
        """步骤 1: 准备 Atlas 先验条件与初始 Mask"""
        logging.info("步骤 1: 准备 Atlas 先验条件与 Mask...")
        path_vol = self._get_atlas_path()
        USE_atlas = False
        condition = svr._initial_mask(
            slices, output_resolution=0.5, device=self.device
        )[0]
        if self.args.dilate:
            condition.dilate_mask_3d(num_iterations=self.args.dilate)
            condition.image = condition.mask.float()
                
        # 暂时保留硬编码的保存路径，后续可提取到 CLI 中
        condition.save('./init_mask.nii.gz', masked=True)
        return condition, path_vol

    def _preprocess_inputs(self, condition: Volume, slices: List[Slice]) -> torch.Tensor:
        """步骤 2: 数据裁剪、缩放与预处理"""
        logging.info("步骤 2: 数据张量变换与预处理...")
        condition_mask = condition.image.float()
        
        condition_mask = label2masks(condition_mask, input_channel=3)
        condition_mask, prefix_shape, nz = resize_img_mask(
            condition_mask, self.batch_size, (self.image_size, self.image_size, self.depth_size)
        )
        condition_mask_tensor = input_transform(condition_mask)
        
        # 将必要参数注入 diffusion 实例 (此处保留原有逻辑)
        self.diffusion.prefix_shape_nz_index = (prefix_shape, nz)
        self.diffusion.slices = slices
        self.diffusion.svr_args = self.args
        self.diffusion.condition_transformation = condition
        
        return condition_mask_tensor

    def _postprocess_and_save(self, img, condition: Volume, condition_mask_tensor: torch.Tensor, slices: List[Slice], path_vol: str):
        """步骤 4: 结果后处理、重采样与保存"""
        logging.info("步骤 4: 结果后处理与 NIfTI 保存...")
        
        if isinstance(img, Volume):
            img.rescale(self.args.output_intensity_mean, masked=True)
            img.save(path=self.args.output_volume, masked=True)
            
            reference_mask = svr._initial_mask(
                slices, output_resolution=0.5, sample_orientation=path_vol, device=self.device
            )[0]
            
            ref_nz, _ = crop_nonzero_tensor(reference_mask.image.cpu())
            img_nz, _, nz = resize_img_mask(img.image.cpu(), self.batch_size, ref_nz.shape)
            mask_nz, _, _ = resize_img_mask(img.mask.cpu(), self.batch_size, ref_nz.shape)
            
            image_reshape = resize_target(img_nz[None, ...], reference_mask.image[None, ...].shape, nz)
            mask_reshape = resize_target(mask_nz[None, ...].float(), reference_mask.image[None, ...].shape, nz)
            
            img_reshape = Volume(
                image=image_reshape[0].to(img.device), mask=mask_reshape[0].bool().to(img.device), 
                transformation=img.transformation,
                resolution_x=img.resolution_x, resolution_y=img.resolution_y, resolution_z=img.resolution_z
            )
            
            reshape_path = self.args.output_volume.replace(".nii.gz", "_reshape.nii.gz")
            img_reshape.save(path=reshape_path, masked=True)
            logging.info(f"重塑(Reshape)后数据已保存至: {reshape_path}")
            
        elif isinstance(img, torch.Tensor):
            img = (img + 1) / 2
            volume = Volume(
                image=img.squeeze(), mask=condition_mask_tensor.squeeze() > 0,
                transformation=condition.transformation, resolution_x=0.5, resolution_y=0.5, resolution_z=0.5
            )
            ablation_path = self.args.output_volume.replace(".nii.gz", "_ablation.nii.gz")
            volume.save(path=ablation_path, masked=True)
            logging.info(f"消融(Ablation)数据已保存至: {ablation_path}")

    def run(self, slices: List[Slice]):
        """流水线主调度入口"""
        logging.info("========== dpmRecon Pipeline 启动 ==========")
        
        # 1. 准备条件先验
        condition, path_vol = self._prepare_condition(slices)
        
        # 2. 数据预处理
        condition_mask_tensor = self._preprocess_inputs(condition, slices)
        
        # 3. 扩散模型推理
        logging.info("步骤 3: 启动扩散模型采样生成...")
        model_inputs = torch.cat([condition_mask_tensor, -1 * torch.ones_like(condition_mask_tensor)], dim=1)
        img = self.diffusion.sample(batch_size=self.batch_size, condition_tensors=model_inputs)
        
        # 4. 结果后处理与保存
        self._postprocess_and_save(img, condition, condition_mask_tensor, slices, path_vol)
        
        logging.info("========== dpmRecon Pipeline 顺利完成 ==========")

# ==========================================
# 批处理报告工具 (Batch Summary Tool)
# ==========================================

def print_batch_summary(mode: str, stats: dict, total: int):
    """打印批处理最终的统计报告"""
    success_count = len(stats["success"])
    failed_count = len(stats["failed"])
    skipped_count = len(stats["skipped"])
    
    logging.info("\n" + "="*50)
    logging.info(f" 批量处理任务完成报告 | 模式: {mode.upper()}")
    logging.info("="*50)
    logging.info(f"总任务数: {total}")
    logging.info(f"✅ 成功: {success_count}")
    logging.info(f"⏭️ 跳过: {skipped_count}")
    logging.info(f"❌ 失败: {failed_count}")
    
    if failed_count > 0:
        logging.error("-" * 50)
        logging.error("【失败的 Subject 列表】:")
        for idx, subj in enumerate(stats["failed"]):
            logging.error(f"  {idx + 1}. {subj}")
    logging.info("="*50 + "\n")
# ==========================================
# 各种执行入口 (Execution Entrypoints)
# ==========================================

def test(global_args: argparse.Namespace):
    # just test
    print(models.USE_TORCH)
    
    # 优雅解析: 直接将命令列表传给 parse_args，不再使用 sys.argv.pop/insert
    nesvor_cmd = ["reconstruct"]
    parser, subparsers = main_parser()
    args = parser.parse_args(nesvor_cmd)
    
    args.dtype = torch.float32 if getattr(args, 'single_precision', False) else torch.float16
    args.timesteps = global_args.timesteps
    args.age = global_args.age
    args.dilate = global_args.dilate
    args.gap = global_args.gap
    args.ablation = ''
    
    input_dict, args = inputs(args)
    slices = _register(args, input_dict['input_stacks'])
    
    DR = dpmRecon(
        args, input_size=128, depth_size=128, num_channels=64, num_class_labels=3, num_res_blocks=1, out_channels=1,
        weightfile=global_args.weight_path
    )
    DR.run(slices)

def main_fide(dataset: DataSet, global_args: argparse.Namespace):
    total_subjects = len(dataset)
    stats = {"success": [], "skipped": [], "failed": []}
    
    logging.info(f"启动 FIDE 批量重建流程，共计 {total_subjects} 个被试 (Subjects).")
    
    for i in range(total_subjects):
        # 安全获取 Subject 名称（根据你原代码的 dataset[i].get_name）
        try:
            subject_name = dataset[i].get_name if hasattr(dataset[i], 'get_name') else f"Subject_{i}"
            if callable(subject_name): subject_name = subject_name()
        except:
            subject_name = f"Subject_{i}"
            
        logging.info(f"\n>>> [进度 {i+1}/{total_subjects}] 开始处理: {subject_name} <<<")
        
        try:
            command = dataset.run(i)
            if command is None:
                logging.warning(f"无需重构，跳过被试: {subject_name}")
                stats["skipped"].append(subject_name)
                continue
                
            nesvor_cmd = ['reconstruct'] + command
            parser, subparsers = main_parser()
            args = parser.parse_args(nesvor_cmd)
            
            # 注入全局参数
            args.device = global_args.device
            args.dtype = torch.float32 if getattr(args, 'single_precision', False) else torch.float16
            args.timesteps = global_args.timesteps
            args.dilate = global_args.dilate
            args.gap = global_args.gap
            args.ablation = ''
            
            current_age = dataset.age(i) if hasattr(dataset, 'age') else 0
            args.age = global_args.age if global_args.age > 0 else current_age

            input_dict, args = inputs(args)

            if getattr(args, 'segmentation', False):
                input_dict['input_stacks'] = _segment_stack(args, input_dict['input_stacks'])
                
            slices, svort_volume = _register(args, input_dict['input_stacks'])
            svort_volume.save(path=args.output_volume.replace(".nii.gz", "_init.nii.gz"))
            
            DR = dpmRecon(
                args, input_size=128, depth_size=128, num_channels=64, num_class_labels=3, num_res_blocks=1, out_channels=1,
                weightfile=global_args.weight_path
            )
            DR.run(slices)
            
            stats["success"].append(subject_name)
            logging.info(f"✅ {subject_name} 重建成功！")
            
        except Exception as e:
            stats["failed"].append(subject_name)
            # exc_info=True 是精髓：它会将完整的错误栈 (Traceback) 打印到日志中！
            logging.error(f"❌ {subject_name} 处理失败: {str(e)}", exc_info=True)
            continue
            
    # 输出最终报告
    print_batch_summary("fide", stats, total_subjects)


def main_nesvor(dataset: DataSet, global_args: argparse.Namespace):
    total_subjects = len(dataset)
    stats = {"success": [], "skipped": [], "failed": []}
    
    logging.info(f"启动 原生 NeSVoR 批量重建流程，共计 {total_subjects} 个被试.")
    
    for i in range(total_subjects):
        try:
            subject_name = dataset[i].get_name if hasattr(dataset[i], 'get_name') else f"Subject_{i}"
            if callable(subject_name): subject_name = subject_name()
        except:
            subject_name = f"Subject_{i}"
            
        logging.info(f"\n>>> [进度 {i+1}/{total_subjects}] 开始处理: {subject_name} <<<")
        
        try:
            command = dataset.run(i)
            if command is None:
                logging.warning(f"无需重构，跳过被试: {subject_name}")
                stats["skipped"].append(subject_name)
                continue
                
            nesvor_cmd = ['reconstruct'] + command
            parser, subparsers = main_parser()
            args = parser.parse_args(nesvor_cmd)
            args.device = global_args.device 
            
            run(args)
            
            stats["success"].append(subject_name)
            logging.info(f"✅ {subject_name} 处理成功！")
            
        except Exception as e:
            stats["failed"].append(subject_name)
            logging.error(f"❌ {subject_name} 处理失败: {str(e)}", exc_info=True)
            continue
            
    print_batch_summary("nesvor", stats, total_subjects)
def sim_dataset_get(sim_in, mode='fide', sim_out=None, mask_folder=None):
    if sim_out is None:
        sim_out = os.path.join(os.path.dirname(sim_in), f'{mode}_noatlas_dr3_result')
    resume = False
    return datesetloader(sim_in, sim_out, mode=mode, resume=resume, maskfolder=mask_folder)

# ==========================================
# 主程序入口 (Main)
# ==========================================
if __name__ == "__main__":
    # 解析命令行参数
    global_args = parse_pipeline_args()
    
    # 构造数据集
    dataset = sim_dataset_get(
        sim_in=global_args.input_dir,
        sim_out=global_args.output_dir,
        mask_folder=global_args.mask_dir,
        mode=global_args.mode
    )
    
    # 根据命令行 flag 设置属性
    if global_args.fbs_seg:
        dataset.FBS_Seg = True
        
    # 分发任务
    if global_args.mode == 'fide':
        main_fide(dataset, global_args)
    elif global_args.mode == 'nesvor':
        main_nesvor(dataset, global_args)
    elif global_args.mode == 'test':
        test(global_args)
    else:
        print(f"Unknown mode: {global_args.mode}")

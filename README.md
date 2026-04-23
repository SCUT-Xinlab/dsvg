# Med-DDPM (Medical Denoising Diffusion Probabilistic Models)

## 📌 项目概述 (Project Overview)
本项目旨在利用扩散模型 (Diffusion Models) 结合 NeSVoR，实现胎儿大脑 MRI 等医疗影像的高质量三维重建、去噪与分割。
项目包含从原始切片 (Slices) 到三维体数据 (Volume) 的完整处理流程，支持引入图谱 (Atlas) 作为先验条件进行扩散模型的引导生成。

## 🌿 分支管理规范 (Branch Management)

为了保证代码的稳定性与研究的可重复性，本项目采用以下分支策略：

*   **`main` (或 `master`) 分支**:
    *   **定位**: 研究基线与稳定版本 (Research Baseline)。
    *   **说明**: 包含已验证跑通的原始实验代码，一般情况下不在此分支直接进行大范围的工程架构修改。
*   **`refactor/production-pipeline` 分支 (当前开发分支)**:
    *   **定位**: 工程化重构与工具化封装分支。
    *   **说明**: 致力于将零散的研究脚本转化为可支持命令行批量调用的稳定模块。消除硬编码，统一日志管理，实现解耦。

### 分支切换指南 (How to switch branches)

如果你需要回到原来的代码跑之前的实验，请使用：
```bash
git checkout main  # 或者 git checkout master
```project
*   **2026/04/23 Phase 1：模块化准备与物理拆分**
    *   创建了 `utils/tensor_ops.py`。
    *   将 `save_tensor_as_nii`, `crop_nonzero_tensor`, `resize_img_mask` 等张量及图像预处理函数从主逻辑中剥离。
    *   重构了主程序 `fidepipe.py` 的包导入顺序（遵循标准库 -> 第三方库 -> 内部模块的 PEP8 规范），并清理了所有的函数级内部 `import`，提高了代码可读性和启动加载规范性。
    *   **Next Step**: 构建纯净的 CLI（命令行接口），去除主程序中暴力修改 `sys.argv` 的不良写法，引入 `argparse` 进行统一参数配置。

*   **2026/04/23 Phase 2.1：引入 JSON 配置支持**
    *   在 `parse_pipeline_args` 中添加了 `--config` 参数支持。
    *   实现了 **JSON文件优先级 > CLI参数 > 默认值** 的层级覆盖逻辑。
    *   允许将长串路径与复杂实验配置固化到 JSON 中（如 `config/default_config.json`），实现 `python fidepipe.py --config xxx.json` 一键启动，极大提升了多组实验的调度效率。

*   **2026/04/23 Phase 3：核心推理类解耦与日志标准化**
    *   引入 Python 标准 `logging` 模块，全面替代原有的 `print`，提供带时间戳和级别的监控日志。
    *   重构 `dpmRecon` 类，遵循单一职责原则 (SoC)。将庞大的 `run()` 方法肢解拆分为 4 个子方法：`_prepare_condition`（图谱准备）、`_preprocess_inputs`（张量缩放处理）、扩散推理、`_postprocess_and_save`（逆向重采样与保存）。
    *   修复了原来代码里加载模型时由于未使用 `map_location` 可能导致的显卡编号错乱问题，以及后处理 `torch.Tensor` 分支时局部变量未绑定的潜在 Bug。
    *   **Next Step**: 实现鲁棒的批处理调度与异常捕获。目前在 `for` 循环批量跑数据时，一个 subject 报错依然只是简单 `continue`，缺乏可追溯的失败队列统计。
    
*   **2026/04/23 Phase 4：重构批处理调度引擎与容错报告**
    *   废弃了原始简单粗暴的 `try-except print(e)` 模式，使用 `logging.error(..., exc_info=True)` 精准捕获并记录完整的错误堆栈 (Traceback)，极大简化了未来排查由于内存溢出 (OOM) 或文件损坏导致的特定数据失败问题。
    *   在批处理循环中引入了 `stats` 字典追踪机制，新增 `print_batch_summary` 方法。在跑完整个数据集后，会在终端（及日志文件）中自动生成可视化的统计报告，明确列出成功、跳过以及失败的被试清单。
    *   增加了进度标识 `[进度 i/Total]`，方便长周期任务的监控。

*   **2026/04/23 Phase 5：扩散模型采样管线重构 (FideGaussianDiffusion)**
    *   重构 `fide_sample_process` 核心流形投影算法，实现核心业务逻辑的区段化（Preprocessing -> Core -> Postprocessing -> Injection）。
    *   将繁杂的数据降维、维度扩充、归一化反转及 `to(device)` 等操作抽离为 `_tensor_to_volume` 和 `_volume_to_tensor` 内部辅助方法。
    *   消除硬编码魔术数字，将诸如 `i % 90 == 0` 的间歇触发阈值抽象为 `gap` 参数。清理冗余及死代码，进一步增强代码的鲁棒性与可测试性。
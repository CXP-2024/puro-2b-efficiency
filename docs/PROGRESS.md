# 进度日志

按日期倒序。每条写清：做了什么、结果是什么、还没验证什么。

## 2026-11-25

**调研**

- 读完选题文档、Puro-2B 论文（网页版，附录部分只读到摘要）和 Puro-Megatron 仓库的 README、文档、启动脚本。
- 确定方向：主线 MoE，副线数据配比。见 [PLAN.md](PLAN.md)。

**集群现状（课程 H100 集群）**

- 8 个节点，每个 8 张 H100，Slurm 调度。每人上限 48 张卡，单作业最长 24 小时。
- 有一个限时 10 小时的 dev 分区。
- 共享存储上有 ClimbMix 已分词数据（约 1.9 TB），Puro 数据尚未下载。
- 直连不了 huggingface.co，hf-mirror.com 和 GitHub 可用。

**发现的缺口**

- 公开仓库只有 2B 全量配置，没有小模型配置、没有评测脚本。
- FP8 blockwise 配方能否在 H100 上跑，未确认。
- 这些问题记在 [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)。

**环境（集群上）**

- 虚拟环境 `~/venvs/puro`，复用课程共享的 Python 3.12 + PyTorch 2.14 (CUDA 13)。用 `source ~/venvs/puro/env.sh` 激活。
- 已装：Puro-Megatron（可编辑安装）、Transformer Engine 2.12.0（仓库锁定的版本，PyTorch 扩展从源码编译）、Emerging-Optimizers v0.2.0（MuonH）。
- CPU 上导入测试全部通过。**GPU 上还没跑过。**
- 踩过的坑：
  - 计算节点没有外网，所有安装只能在登录节点做。
  - 登录节点没有 CUDA 工具链。用 pip 的 `nvidia-cuda-nvcc`、`nvidia-cuda-crt`、`nvidia-cuda-cccl` 加共享环境里的头文件，拼出 `~/venvs/puro/cuda_home` 才编过 Transformer Engine。
  - PyPI 官方源超时，改用阿里云镜像。GitHub 从集群访问不稳定，Emerging-Optimizers 是本机克隆后传上去的。
  - Transformer Engine 运行时要设 `CUDNN_PATH`，否则报找不到 cuDNN。

**数据**

- 通过 hf-mirror 下载了 `phase1/fineweb-edu-dedup` 的前 4 个 Parquet 文件和分词器，共 9.0 GB。
- 训练代码读的格式：一个 `metadata.json` 加若干 `.npy`，每行 `序列长度 + 1` 个 token，文档边界由 EOS（151645）在运行时识别。
- 写了转换脚本 [scripts/pack_parquet_to_npy.py](../scripts/pack_parquet_to_npy.py)。试转了 1.54 亿 token（8 线程约 190 万 token/秒）。
- 我们的打包方式（文档首尾相接后按固定长度切行、保持原始顺序）是自己选的，不保证与作者一致。只影响与论文绝对 loss 的对比。

**训练脚本**

- [scripts/run_proxy.sh](../scripts/run_proxy.sh)：照 Puro-2B 的 phase1 配方缩小，模型大小、batch、精度、GPU 数用环境变量控制。
- [scripts/smoke.sbatch](../scripts/smoke.sbatch)：导入测试，然后一个极小模型分别用 BF16 和 FP8 跑 30 步。已提交，排队中。

**Proposal**

- 3 页 Beamer 初稿在 [proposal/main.tex](../proposal/main.tex)，关键数字已对照论文原文核对。
- 待办：填队员姓名；放进 Overleaf。

**下一步**

- 看冒烟测试结果，特别是 FP8 blockwise 能否在 H100 上跑。
- 把 4 个 Parquet 全量转换，跑 E0 基线。

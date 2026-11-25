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
- FP8 blockwise 配方能否在 H100 上跑：当时未确认，后来冒烟测试验证了能跑（见下）。
- 这些问题记在 [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)。

**环境（集群上）**

- 虚拟环境 `~/venvs/puro`，复用课程共享的 Python 3.12 + PyTorch 2.14 (CUDA 13)。用 `source ~/venvs/puro/env.sh` 激活。
- 已装：Puro-Megatron（可编辑安装）、Transformer Engine 2.12.0（仓库锁定的版本，PyTorch 扩展从源码编译）、Emerging-Optimizers v0.2.0（MuonH）。
- CPU 和 GPU（H100）上导入测试全部通过。
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
- [scripts/smoke.sbatch](../scripts/smoke.sbatch)：导入测试，然后一个极小模型分别用 BF16 和 FP8 跑 30 步。
- [scripts/env.sh](../scripts/env.sh)：集群上环境脚本的副本。

**冒烟测试结果（1 张 H100，8 层、hidden 512，batch 8，序列长 4096，30 步）**

| 精度 | 训练 loss（第 1 步 → 第 30 步） | 验证 loss | 每步耗时 | 吞吐 |
|---|---|---|---|---|
| BF16 | 12.01 → 6.75 | 6.94 | 约 183 ms | 约 128 TFLOP/s |
| FP8 blockwise | 12.01 → 6.74 | 6.94 | 约 214 ms | 约 110 TFLOP/s |

- 训练代码、MuonH、我们自己打包的数据，端到端跑通。
- **FP8 blockwise 配方能在 H100 上跑**，但在这个极小模型上比 BF16 慢约 17%。H100 上的代理实验先用 BF16。
- 只有 30 步、一个极小模型，这些数字只说明流程通了，不代表基线质量。

**跑通前修的三个问题**

- 虚拟环境里没有 `torchrun`，改用 `python -m torch.distributed.run`。
- 输出层的梯度累加融合需要 Apex 的 CUDA 扩展，我们没装，用 `--no-gradient-accumulation-fusion` 关掉。作者的环境装了 Apex，这是一处与原配方的差异（影响速度和显存，不影响数值）。
- 计算节点系统里有 CUDA 12 运行库，Transformer Engine 发现 `libcudart.so.12` 和 `.13` 都能加载就拒绝运行。在库搜索路径最前面放一个同名空文件，让系统那份加载不了。

**Proposal**

- 3 页 Beamer 初稿在 [proposal/main.tex](../proposal/main.tex)，关键数字已对照论文原文核对。
- 待办：填队员姓名；放进 Overleaf。

**下一步**

- 把 4 个 Parquet 全量转换，跑 E0 基线。
- 定 E0 的具体规模和 token 数（取决于再下多少数据）。

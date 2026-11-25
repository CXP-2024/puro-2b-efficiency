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

**下一步**

- 在集群上搭 Puro-Megatron 环境，跑小模型冒烟测试。

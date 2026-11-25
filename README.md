# Improving the Compute Efficiency of Puro-2B

课程项目（LLMs from Scratch, Fall 2026 — Speedrun Topic 1）。

**目标**：用少 30% 的算力，达到 [Puro-2B](https://arxiv.org/abs/2608.27370) 在 15 个下游 benchmark 上的平均分。

| | 基线（论文正式版本） | 本项目目标 |
|---|---|---|
| 15-benchmark 平均分 | 57.81 | ≥ 57.81 |
| 算力（RTX 5090 GPU 小时） | 22,514 | ≤ 15,760 |

> 表中数字转述自论文网页版，写进 proposal 前需对照原文核对。

## 文档

- [docs/PLAN.md](docs/PLAN.md) — 方向、实验协议、第一个实验
- [docs/PROGRESS.md](docs/PROGRESS.md) — 进度日志（按日期倒序）
- [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) — 需要向助教 / Puro 作者确认的问题

## 目录

```
docs/      计划、进度、待确认问题
configs/   小模型代理实验的配置
scripts/   集群上的环境搭建、作业提交脚本
results/   实验结果（曲线数据、汇总表；不放 checkpoint）
```

## 上游

- 训练代码：[thu-pacman/Puro-Megatron](https://github.com/thu-pacman/Puro-Megatron)（基于 Megatron-LM core_v0.16.0）
- 数据：[thu-pacman/Puro-2B](https://huggingface.co/datasets/thu-pacman/Puro-2B)
- 模型：[thu-pacman/Puro-2B-Base](https://huggingface.co/thu-pacman/Puro-2B-Base)

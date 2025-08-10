# 大模型评估框架

[English Version](README_en.md)

> 本 README 由 Qwen Code 生成

实现一个大模型评估框架，支持大模型安全、能力和拒绝回答的评估，能够统一进行模型安全攻击和防御的评估。

## 功能特性

- 安全性评估，支持多种攻击方法
- 能力评估，支持多个基准测试
- 支持基于API和本地模型推理
- 可配置的评估流程
- 缓存管理，提高重复评估效率
- 指标计算，用于性能分析

## 环境配置

### 前置要求

- Python 3.12 或更高版本
- [uv](https://github.com/astral-sh/uv) 用于依赖管理
- Redis 服务器（用于缓存，可选）
- GPU 资源（用于本地模型推理，可选）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yibotongxue/llm-evaluator
   cd llm-evaluator
   ```

2. 使用 uv 安装依赖：
   ```bash
   uv sync
   ```

3. 设置环境变量：
   导出 Qwen API 密钥（此阶段 API 推理必需）：
   ```bash
   export QWEN_API_KEY=your_qwen_api_key_here
   ```

## 运行评估器

### 基本用法

使用特定配置运行安全性评估：

```bash
QWEN_API_KEY=your_qwen_api_key_here uv run -m llm_evaluator.benchmark.benchmark --config-file-path ./configs/safety.yaml
```

### 配置说明

框架使用 YAML 配置文件来定义：
- 模型设置（本地或基于API）
- 评估基准测试
- 攻击方法
- 指标计算
- 缓存设置

示例配置文件可在 `configs/` 目录中找到。

### 可用基准测试

框架支持多个基准测试，包括：
- AdvBench
- StrongReject
- JBB-Behaviors
- WildJailbreak（Vanilla 和 Adversarial）

### 攻击方法

实现了多种基于提示词的攻击方法：
- Base64Attack
- AIMAttack
- Dev Mode v2
- BetterDAN
- 以及其他多种方法

## 自定义扩展

要自定义评估：

1. 在 `configs/` 目录中修改或创建新的配置文件
2. 在 `llm_evaluator/prompts/attack/` 目录中添加新的攻击方法
3. 在 `llm_evaluator/metrics/` 目录中实现新的指标
4. 在 `llm_evaluator/data/` 目录中添加新的数据格式

## 输出结果

评估结果保存在配置文件中指定的输出目录中，通常包括：
- 完整结果及详细指标
- 简要摘要结果
- 配置备份

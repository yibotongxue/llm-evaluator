# 大模型评估

实现一个大模型评估框架，支持大模型安全、能力和过渡拒绝的评估，以统一的进行模型安全攻击和防御的评估。

## 运行

安装依赖：

```bash
uv sync
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121 --no-build-isolation
```

运行：

```bash
QWEN_API_KEY=你的API密钥 uv run -m llm_evaluator.benchmark.safety --config-file-path ./configs/evaluate.yaml
```

# vLLM Code Review Instructions

本仓库为 **vLLM** —— 高性能 LLM 推理与服务框架，主要技术栈为 Python + C++/CUDA。
进行代码审查或生成代码建议时，请遵循以下规则。

## 1. 代码风格

- Python 代码必须通过仓库 pre-commit 链路：`ruff`、`yapf`、`isort`、`mypy`、`codespell`
  （配置见 `.pre-commit-config.yaml`、`pyproject.toml`）。
- 所有新增 Python 文件首行必须包含 SPDX 许可证声明：
  `# SPDX-License-Identifier: Apache-2.0`
- 公开 API（函数 / 类 / 方法）必须包含类型注解（Type Hints）与 docstring。
- C++ / CUDA 代码遵循 `.clang-format`，禁止顶层 `using namespace`，头文件使用 `#pragma once`。
- 禁止 `print` 调试代码，统一使用 `logger = init_logger(__name__)`。

## 2. 提交与 PR 规范

- PR 标题必须带分类前缀，如：`[Bugfix]`、`[Feature]`、`[Model]`、`[Kernel]`、
  `[CI/Build]`、`[Doc]`、`[Frontend]`、`[Misc]`。
- 每个 commit 必须含 DCO 签名（`Signed-off-by:`）。
- 单个 PR 聚焦单一目标，避免捆绑无关改动；大重构应拆分为多个 PR。

## 3. 测试要求

- 新功能必须在 `tests/` 下提供对应测试：
    - 模型支持：`tests/models/`
    - 算子 / Kernel：`tests/kernels/`
    - 引擎 / 调度：`tests/engine/`、`tests/v1/`
    - 服务接口：`tests/entrypoints/`
- 修复 bug 须附带能复现该 bug 的回归测试。
- 改动 `vllm/engine/`、`vllm/worker/`、`vllm/v1/` 等核心路径必须有测试覆盖。

## 4. 性能与正确性

- CUDA Kernel / 关键算子改动必须在 PR 描述中提供基准（benchmark）对比数据。
- 禁止在推理热路径引入 Python 级循环；优先使用 PyTorch 张量化操作或自定义 kernel。
- 涉及多卡 / 分布式的改动须在 PR 描述中说明对 TP / PP / DP / EP 的兼容性。
- 内存敏感路径（KV cache、PagedAttention）的改动须评估显存占用变化。

## 5. 安全

- 严禁在代码中硬编码任何 token、API key、私有 endpoint、内部地址。
- 不得引入未经评审的远程下载（`curl | sh`、未 pin 版本的依赖等）。
- 新增依赖必须同步更新 `requirements/*.txt` 对应文件，并在 PR 描述中说明引入原因与替代方案评估。

## 6. 兼容性

- 改动公开 API（`vllm/__init__.py` 暴露的符号、HTTP / OpenAI-compatible 接口）须在 PR 描述中标注 breaking change 并给出迁移说明。
- 不得在不更新文档的前提下变更 CLI 参数或环境变量语义。

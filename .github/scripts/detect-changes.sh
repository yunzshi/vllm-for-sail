#!/usr/bin/env bash
# ==============================================================================
# Fork CI - Change Detection Script
# ==============================================================================
# Detects which test workflow groups need to run based on git diff.
# Reads source_file_dependencies patterns from .buildkite/test_areas/*.yaml
# and outputs GitHub Actions step outputs: run_core, run_kernels, etc.
#
# Usage: detect-changes.sh [base_ref]
#   base_ref: git ref to compare against (default: origin/main)
#
# Groups → Test Areas mapping:
#   core         → attention, basic_correctness, cuda, engine, model_executor
#   kernels      → kernels
#   compile      → compile
#   distributed  → distributed, expert_parallelism
#   entrypoints  → entrypoints
#   models       → models_language, models_multimodal, models_basic, models_distributed
#   functional   → lora, quantization, samplers, pytorch, plugins, ray_compat, weight_loading
#   misc         → misc, benchmarks, lm_eval, e2e_integration
# ==============================================================================

set -euo pipefail

BASE_REF="${1:-origin/main}"
ALL_GROUPS=(core kernels compile distributed entrypoints models functional misc)

# ==============================================================================
# Get list of changed files
# ==============================================================================
CHANGED_FILES=$(git diff --name-only "${BASE_REF}"...HEAD 2>/dev/null || \
                git diff --name-only "${BASE_REF}" HEAD 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
  echo "::notice::No changed files detected. Skipping all tests."
  for group in "${ALL_GROUPS[@]}"; do
    echo "run_${group}=false" >> "${GITHUB_OUTPUT:-/dev/stdout}"
  done
  echo "run_all=false" >> "${GITHUB_OUTPUT:-/dev/stdout}"
  exit 0
fi

FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
echo "Detected ${FILE_COUNT} changed file(s)."
echo "$CHANGED_FILES" | head -30
[ "$FILE_COUNT" -gt 30 ] && echo "... and $((FILE_COUNT - 30)) more"

OUTPUT_FILE="${GITHUB_OUTPUT:-/dev/stdout}"

# ==============================================================================
# Helper: prefix-based path matching
# ==============================================================================
file_matches() {
  local file="$1" pattern="$2"
  [[ "$file" == "$pattern" || "$file" == "${pattern}"* ]]
}

# ==============================================================================
# Global run_all check (from .buildkite/ci_config.yaml)
# ==============================================================================
RUN_ALL=false

RUN_ALL_PATTERNS=(
  "docker/Dockerfile"
  "CMakeLists.txt"
  "requirements/common.txt"
  "requirements/cuda.txt"
  "requirements/build.txt"
  "requirements/test.txt"
  "setup.py"
  "csrc/"
  "cmake/"
  ".github/workflows/fork-ci"
  ".github/scripts/detect-changes.sh"
)

RUN_ALL_EXCLUDE=(
  "docker/Dockerfile."
  "csrc/cpu/"
  "csrc/rocm/"
  "cmake/hipify.py"
  "cmake/cpu_extension.cmake"
)

while IFS= read -r file; do
  matched=false
  for p in "${RUN_ALL_PATTERNS[@]}"; do
    if file_matches "$file" "$p"; then matched=true; break; fi
  done
  [ "$matched" = false ] && continue

  excluded=false
  for ex in "${RUN_ALL_EXCLUDE[@]}"; do
    if file_matches "$file" "$ex"; then excluded=true; break; fi
  done

  if [ "$excluded" = false ]; then
    RUN_ALL=true
    echo "::notice::run_all triggered by changed file: $file"
    break
  fi
done <<< "$CHANGED_FILES"

echo "run_all=${RUN_ALL}" >> "$OUTPUT_FILE"

# ==============================================================================
# Group check function
# ==============================================================================
check_group() {
  local name="$1"; shift
  local patterns=("$@")

  if [ "$RUN_ALL" = true ]; then
    echo "run_${name}=true" >> "$OUTPUT_FILE"
    echo "  ✓ ${name} (run_all)"
    return
  fi

  while IFS= read -r file; do
    for p in "${patterns[@]}"; do
      if file_matches "$file" "$p"; then
        echo "run_${name}=true" >> "$OUTPUT_FILE"
        echo "  ✓ ${name} (matched: ${file})"
        return
      fi
    done
  done <<< "$CHANGED_FILES"

  echo "run_${name}=false" >> "$OUTPUT_FILE"
  echo "  ✗ ${name}"
}

echo ""
echo "=== Test Group Detection ==="

# ==============================================================================
# Group: core
# Test areas: attention, basic_correctness, cuda, engine, model_executor
# ==============================================================================
check_group "core" \
  "vllm/" \
  "tests/v1/attention" \
  "tests/v1/cudagraph" \
  "tests/v1/e2e" \
  "tests/v1" \
  "tests/basic_correctness/" \
  "tests/cuda" \
  "tests/engine" \
  "tests/test_sequence" \
  "tests/test_config" \
  "tests/test_logger" \
  "tests/test_vllm_port" \
  "tests/model_executor" \
  "tests/entrypoints/openai/test_tensorizer_entrypoint.py"

# ==============================================================================
# Group: kernels
# Test areas: kernels
# ==============================================================================
check_group "kernels" \
  "csrc/" \
  "vllm/model_executor/layers/attention" \
  "vllm/model_executor/layers/quantization" \
  "vllm/model_executor/layers/mamba/ops" \
  "vllm/model_executor/layers/fused_moe" \
  "vllm/v1/attention" \
  "vllm/distributed/device_communicators/" \
  "vllm/envs.py" \
  "vllm/config" \
  "vllm/platforms/cuda.py" \
  "vllm/utils/deep_gemm.py" \
  "vllm/utils/import_utils.py" \
  "tools/install_deepgemm.sh" \
  "tests/kernels/" \
  "tests/models/quantization/test_nvfp4.py"

# ==============================================================================
# Group: compile
# Test areas: compile
# ==============================================================================
check_group "compile" \
  "vllm/model_executor/" \
  "vllm/compilation/" \
  "vllm/v1/worker/" \
  "vllm/v1/cudagraph_dispatcher.py" \
  "vllm/v1/attention/" \
  "csrc/quantization/" \
  "tests/compile/"

# ==============================================================================
# Group: distributed
# Test areas: distributed, expert_parallelism
# ==============================================================================
check_group "distributed" \
  "vllm/" \
  "tests/distributed" \
  "tests/v1/distributed" \
  "tests/v1/shutdown" \
  "tests/v1/entrypoints/openai/test_multi_api_servers.py" \
  "tests/v1/worker/test_worker_memory_snapshot.py" \
  "tests/v1/engine/test_engine_core_client.py" \
  "tests/v1/kv_connector/" \
  "tests/compile/fullgraph/test_basic_correctness.py" \
  "tests/compile/test_wrapper.py" \
  "tests/entrypoints/llm/test_collective_rpc.py" \
  "tests/examples/" \
  "examples/offline_inference/"

# ==============================================================================
# Group: entrypoints
# Test areas: entrypoints
# ==============================================================================
check_group "entrypoints" \
  "vllm/" \
  "csrc/" \
  "tests/entrypoints/" \
  "tests/v1" \
  "tests/tool_use" \
  "tests/v1/entrypoints"

# ==============================================================================
# Group: models
# Test areas: models_language, models_multimodal, models_basic, models_distributed
# ==============================================================================
check_group "models" \
  "vllm/" \
  "tests/models/" \
  "tests/basic_correctness/" \
  "tests/model_executor/model_loader/test_sharded_state_loader.py"

# ==============================================================================
# Group: functional
# Test areas: lora, quantization, samplers, pytorch, plugins, ray_compat,
#             weight_loading
# ==============================================================================
check_group "functional" \
  "vllm/" \
  "csrc/" \
  "requirements/" \
  "setup.py" \
  "tests/lora" \
  "tests/quantization" \
  "tests/models/quantization" \
  "tests/samplers" \
  "tests/conftest.py" \
  "tests/compile" \
  "tests/plugins/" \
  "tests/plugins_tests/" \
  "tests/distributed/test_distributed_oot.py" \
  "tests/entrypoints/openai/test_oot_registration.py" \
  "tests/models/test_oot_registration.py" \
  "tests/weight_loading"

# ==============================================================================
# Group: misc
# Test areas: misc, benchmarks, lm_eval, e2e_integration
# ==============================================================================
check_group "misc" \
  "vllm/" \
  "csrc/" \
  "benchmarks/" \
  "examples/" \
  "setup.py" \
  "tests/v1" \
  "tests/test_regression" \
  "tests/detokenizer" \
  "tests/multimodal" \
  "tests/utils_" \
  "tests/test_inputs.py" \
  "tests/test_outputs.py" \
  "tests/test_pooling_params.py" \
  "tests/test_ray_env.py" \
  "tests/renderers" \
  "tests/standalone_tests/" \
  "tests/tokenizers_" \
  "tests/tool_parsers" \
  "tests/transformers_utils" \
  "tests/config" \
  "tests/benchmarks/" \
  "tests/evals/"

echo ""
echo "=== Detection Complete ==="

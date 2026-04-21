# PLaMo2 NPUW LLM Change Report (2026-04-21)

## Scope

- Repository: `libs/openvino`
- Branch: `shunta/plamo2-npu-work-20260413`
- Area: `src/plugins/intel_npu/src/plugin/npuw`
- Changed files: 9

## Executive Summary

This change set turns the NPUW LLM path into a much more observable runtime by adding fine-grained profiling around prefill, generate, KV-cache copy/update, and subgraph execution. In parallel, it hardens several PLaMo2-specific data paths, especially for Mamba-style external state tensors that do not behave like ordinary attention KV-cache tensors.

The most important behavioral change is that `LLMInferRequest::bind_past_kv()` is now effectively disabled with an early return. The rest of the patch still supports KV/state propagation, but does so via explicit copy/update flows that are easier to profile and safer for mixed tensor layouts. That tradeoff likely reduces aliasing-related risk at the cost of additional copy overhead.

## Detailed Changes

### 1. Thread-safe runtime profiling was added to the generic NPUW request path

Files:

- `base_sync_infer_request.cpp`
- `base_sync_infer_request.hpp`
- `just_sync_infer_request.cpp`

Highlights:

- Added `record_profile_metric()` helper guarded by `m_profile_mutex` so profile aggregation is safe when timing data is collected from multiple execution paths.
- Instrumented `IBaseInferRequest::infer()` around:
  - `prepare_for_infer`
  - `subscribe_subrequest`
  - `run_subrequest_for_success`
  - per-subgraph execution (`infer/subgraph_<idx>`)
  - `complete_subrequest`
  - accuracy checks
- Instrumented `JustInferRequest` preparation and execution phases, including:
  - pyramid selector/request preparation
  - first parameter binding
  - function-call head prefill
  - spatial / dynamic attention / HFA preparation
  - function prologue input/output binding
  - closure unpacking
  - overlapped async execution/wait sections
  - specialized infer paths for spatial, HFA, and MoE

Effect:

- The patch makes it possible to attribute runtime cost to concrete pipeline steps instead of only coarse request-level timing.

### 2. LLM runtime profiling was added for prefill/generate/KV-cache handling

Files:

- `llm_infer_base_request.cpp`
- `llm_infer_base_request.hpp`
- `llm_infer_request.cpp`
- `llm_infer_request.hpp`

Highlights:

- Added `RuntimeProfile` plumbing to `LLMInferBaseRequest` through:
  - `bind_runtime_profile()`
  - `record_runtime_metric(scope, step, fn)`
- `update_kvcache_for()` now accepts an optional `profile_scope` and records:
  - tensor lookup
  - slice construction
  - full and partial copy paths
  - dedicated Mamba-state copy steps
- `LLMInferRequest` now owns `m_llm_profile`, binds it at construction time, and exposes combined profiling through `get_profiling_info()`.
- Prefill/generate execution was instrumented across:
  - conversation reset
  - variant selection
  - LoRA application
  - longrope handling
  - input preparation
  - infer calls
  - LM head execution
  - logits extraction
  - hidden-state updates
  - chunked prefill loops
  - KV-cache copy/update

Effect:

- Runtime cost can now be analyzed at stage and sub-step granularity for both prefill and generate.

### 3. Operator profiling summaries were added for debugging hot subgraphs

Files:

- `llm_infer_request.cpp`
- `llm_infer_request.hpp`

Highlights:

- Added `OpProfileSummary` aggregation for executed nodes.
- Added `collect_stage_profiling()` to accumulate OpenVINO operator profiling records after prefill/generate inference.
- Added destructor reporting that prints:
  - top hot operators per stage
  - hottest operator per subgraph
- Added `get_profiling_info()` override that merges stage-local profiling info with stage name prefixes.

Effect:

- When profiling is enabled, this patch provides an immediate hot-operator summary without requiring external post-processing.

### 4. PLaMo2 Mamba external state handling was tightened

Files:

- `llm_compiled_model.cpp`
- `llm_infer_base_request.cpp`
- `llm_infer_request.cpp`

Highlights:

- Introduced reusable `is_plamo2_mamba_external_state(...)` logic in `llm_compiled_model.cpp`.
- `cvt_kvcache_to_low_precision()` now skips low-precision conversion entirely when a model contains PLaMo2 Mamba external state tensors.
- KV/state copy logic now treats Mamba tensors as a special case:
  - full copy when shapes match
  - bounded slice copy when shapes differ

Effect:

- This avoids applying ordinary attention-KV assumptions to Mamba state tensors and reduces the chance of corrupting large state buffers.

### 5. Prefill-to-generate buffer sharing was intentionally short-circuited

File:

- `llm_infer_request.cpp`

Highlights:

- `LLMInferRequest::bind_past_kv()` now returns immediately before any tensor rebinding happens.
- The old rebinding logic remains in place but is unreachable.

Interpretation:

- The patch appears to prefer explicit KV/state copy/update over direct tensor aliasing between prefill and generate requests.
- This is consistent with the rest of the patch, which adds profiling and special-case handling around explicit copy paths.

Operational note:

- This likely improves safety and debuggability, but it may regress latency or memory bandwidth compared with shared-buffer execution.

### 6. Profiling can now be enabled outside developer-only builds

File:

- `logging.cpp`

Highlights:

- Removed the `NPU_PLUGIN_DEVELOPER_BUILD` guard around environment-variable based profiling enablement.

Effect:

- Runtime profiling can now be activated in more build configurations, which matches the new profiling-heavy instrumentation added in this patch.

## Expected Intent

The patch appears to be aimed at answering two concrete questions:

1. Where does time go inside NPUW LLM prefill/generate execution, especially across subgraphs and KV-cache transitions?
2. How can PLaMo2 Mamba-specific state tensors be handled without relying on assumptions that only hold for standard attention KV-cache layouts?

The instrumentation and the `bind_past_kv()` short-circuit strongly suggest this branch is being used for root-cause analysis and performance investigation rather than for a final throughput-optimized implementation.

## Validation Status

Completed:

- `git diff --check` on `libs/openvino` completed without whitespace errors.

Not completed in the current shell:

- CMake/CTest-based binary tests were not run because `cmake` and `ctest` are not available on the current PATH in this environment.

## Follow-up Recommendations

- If the early return in `bind_past_kv()` is intended only for investigation, replace it with a guarded runtime switch before merging broadly.
- Add a targeted regression test for PLaMo2 Mamba external-state copy behavior, especially for mismatched source/destination sequence lengths.
- Capture before/after latency for:
  - prefill
  - first generate step
  - steady-state generate step
  - chunked prefill with `m_past_kv_bound` enabled and disabled

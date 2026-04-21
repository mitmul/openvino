// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <string_view>

#include "llm_compiled_model.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "perf.hpp"

namespace ov {
namespace npuw {

class LLMInferBaseRequest : public ov::ISyncInferRequest {
public:
    using RuntimeMetric = ov::npuw::perf::metric<ov::npuw::perf::MSec>;
    using RuntimeProfile = ov::npuw::perf::Profile<RuntimeMetric>;

    struct layer_names {
        static constexpr const char* input_ids = "input_ids";
        static constexpr const char* inputs_embeds = "inputs_embeds";
        static constexpr const char* attention_mask = "attention_mask";
        static constexpr const char* position_ids = "position_ids";
        static constexpr const char* past_key_values = "past_key_values";
        static constexpr const char* output_embeds = "npuw_output_embed";
        static constexpr const char* logits = "logits";
        static constexpr const char* token_type_ids = "token_type_ids";
        static constexpr const char* longrope_input = "npuw_longrope_input";
    };

    struct layer_ids {
        static constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;
        static constexpr std::size_t kStartOutputKVCacheLayers = 1;
    };

    using PortsMap = std::unordered_map<std::string, ov::Output<const ov::Node>>;

    explicit LLMInferBaseRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model)
        : ISyncInferRequest(compiled_model),
          m_npuw_llm_compiled_model(compiled_model) {}

    void check_tensors() const override {};
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

protected:
    void bind_runtime_profile(RuntimeProfile* runtime_profile) {
        m_runtime_profile = runtime_profile;
    }

    template <typename F>
    void record_runtime_metric(std::string_view scope, std::string_view step, F&& f) {
        if (m_runtime_profile == nullptr || scope.empty() || step.empty()) {
            f();
            return;
        }

        auto sample = ov::npuw::perf::MSec::sample(std::forward<F>(f));
        std::string tag;
        tag.reserve(scope.size() + step.size() + 1);
        tag.append(scope);
        tag.push_back('/');
        tag.append(step);
        std::lock_guard lock(m_runtime_profile_mutex);
        (*m_runtime_profile)[tag] += std::move(sample);
    }

    void update_kvcache_for(std::shared_ptr<ov::IAsyncInferRequest> request,
                            const PortsMap& in_ports,
                            const PortsMap& out_ports,
                            uint32_t num_tokens,
                            bool v_transposed,
                            std::string_view profile_scope = {});
    void init_tensor(const ov::Output<const ov::Node>& port);
    void init_ports();

protected:
    std::shared_ptr<LLMCompiledModel> m_npuw_llm_compiled_model;
    RuntimeProfile* m_runtime_profile = nullptr;
    std::mutex m_runtime_profile_mutex;
};

}  // namespace npuw
}  // namespace ov

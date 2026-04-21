// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_base_request.hpp"

#include <regex>

#include "infer_request_utils.hpp"

namespace {
bool is_plamo2_mamba_state_tensor(const ov::SoPtr<ov::ITensor>& tensor) {
    const auto& shape = tensor->get_shape();
    return shape.size() == 4u && shape[1] == 1u && shape[3] > 1024u;
}
}  // namespace

void ov::npuw::LLMInferBaseRequest::update_kvcache_for(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t num_tokens,
    bool v_transposed,
    std::string_view profile_scope) {
    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    auto& compiled = request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = layer_ids::kStartOutputKVCacheLayers; i < compiled->outputs().size(); ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
        if (in_ports.find(input_name) == in_ports.end()) {
            // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }
        ov::SoPtr<ov::ITensor> dst_tensor;
        ov::SoPtr<ov::ITensor> src_tensor;
        record_runtime_metric(profile_scope, "update_kvcache_tensor_lookup", [&]() {
            dst_tensor = request->get_tensor(in_ports.at(input_name));
            src_tensor = request->get_tensor(out_ports.at(output_name));
        });
        const auto& kv_dim = (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kvcache_desc.dim;

        if (is_plamo2_mamba_state_tensor(src_tensor) && is_plamo2_mamba_state_tensor(dst_tensor)) {
            if (src_tensor->get_shape() == dst_tensor->get_shape()) {
                record_runtime_metric(profile_scope, "update_kvcache_mamba_copy_full", [&]() {
                    src_tensor->copy_to(dst_tensor._ptr);
                });
            } else {
                const auto common_seq =
                    static_cast<uint32_t>(std::min(src_tensor->get_shape()[kv_dim], dst_tensor->get_shape()[kv_dim]));
                auto src_slice = uu::make_tensor_slice(src_tensor, kv_dim, 0u, common_seq);
                auto dst_slice = uu::make_tensor_slice(dst_tensor, kv_dim, 0u, common_seq);
                record_runtime_metric(profile_scope, "update_kvcache_mamba_copy_slice", [&]() {
                    uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
                });
            }
            continue;
        }

        ov::SoPtr<ov::ITensor> dst_slice;
        record_runtime_metric(profile_scope, "update_kvcache_make_dst_slice", [&]() {
            dst_slice = uu::make_tensor_slice(dst_tensor,
                                              kv_dim,
                                              kvcache_desc.num_stored_tokens - num_tokens,
                                              kvcache_desc.num_stored_tokens);
        });

        // NOTE: Sometimes present kv layer can contain greater seq_len
        //       than was sent to be processed
        uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
        OPENVINO_ASSERT(num_tokens <= src_seq_len);
        if (src_seq_len > num_tokens) {
            ov::SoPtr<ov::ITensor> src_slice;
            record_runtime_metric(profile_scope, "update_kvcache_make_src_slice", [&]() {
                src_slice = uu::make_tensor_slice(src_tensor, kv_dim, src_seq_len - num_tokens, src_seq_len);
            });
            record_runtime_metric(profile_scope, "update_kvcache_copy_tail", [&]() {
                uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
            });
        } else {
            record_runtime_metric(profile_scope, "update_kvcache_copy_full", [&]() {
                uu::copy_tensor_by_dim(src_tensor, dst_slice, kv_dim, kv_dim);
            });
        }
    }
}

void ov::npuw::LLMInferBaseRequest::init_tensor(const ov::Output<const ov::Node>& port) {
    ov::SoPtr<ITensor> tensor;
    tensor = ov::ISyncInferRequest::get_tensor(port);

    if (!tensor) {
        const auto& shape = port.get_partial_shape();
        const bool is_dynamic = shape.is_dynamic();
        ov::Shape tensor_shape;
        if (is_dynamic) {
            for (auto&& item : shape) {
                tensor_shape.push_back(item.is_static() ? item.get_length() : 0);
            }
        } else {
            tensor_shape = shape.to_shape();
        }

        tensor = ov::make_tensor(port.get_element_type(), tensor_shape);
        set_tensor(port, tensor);
    }
}

void ov::npuw::LLMInferBaseRequest::init_ports() {
    for (const auto& input_port : m_npuw_llm_compiled_model->inputs()) {
        init_tensor(input_port);
    }
    for (const auto& output_port : m_npuw_llm_compiled_model->outputs()) {
        init_tensor(output_port);
    }
}

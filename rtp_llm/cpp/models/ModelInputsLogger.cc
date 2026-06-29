#include "rtp_llm/cpp/models/ModelInputsLogger.h"

#include <ATen/core/Dict.h>
#include <ATen/core/jit_type.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <torch/serialize.h>

namespace rtp_llm {

namespace {

std::string timestampForFile(int64_t now_us) {
    std::ostringstream us;
    us << std::setfill('0') << std::setw(6) << (now_us % 1000000);
    return autil::TimeUtility::usFormat(now_us, "%Y%m%d_%H%M%S") + "_" + us.str();
}

torch::Tensor tensorForDump(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return {};
    }
    auto out = tensor.detach();
    if (out.scalar_type() == torch::kFloat8_e4m3fn) {
        out = out.view(torch::kChar);
    }
    out = out.contiguous();
    if (out.is_cuda()) {
        out = out.cpu();
    }
    return out;
}

void addTensor(c10::impl::GenericDict& payload,
               const std::string&      name,
               const torch::Tensor&    tensor,
               const torch::Tensor&    host_snapshot = {}) {
    const auto& source = host_snapshot.defined() ? host_snapshot : tensor;
    if (!source.defined()) {
        payload.insert(name, c10::IValue());
        return;
    }
    payload.insert(name, tensorForDump(source));
}

void addSize(c10::impl::GenericDict& payload, const std::string& name, size_t value) {
    payload.insert(name, static_cast<int64_t>(value));
}

c10::impl::GenericDict buildModelInputsPayload(const GptModelInputs& inputs) {
    c10::impl::GenericDict payload(c10::StringType::get(), c10::AnyType::get());
    payload.reserve(96);

    payload.insert("trace_ids", inputs.trace_ids);

    addTensor(payload, "combo_tokens", inputs.combo_tokens, inputs.combo_tokens_host_for_log);
    addTensor(payload, "input_lengths", inputs.input_lengths, inputs.input_lengths_host_for_log);
    addTensor(payload, "sequence_lengths", inputs.sequence_lengths, inputs.sequence_lengths_host_for_log);
    addTensor(payload, "lm_output_indexes", inputs.lm_output_indexes);
    addTensor(payload, "lm_output_lengths", inputs.lm_output_lengths);
    addTensor(payload, "prefix_lengths", inputs.prefix_lengths, inputs.prefix_lengths_host_for_log);
    addTensor(payload, "combo_tokens_type_ids", inputs.combo_tokens_type_ids);
    addTensor(payload, "combo_position_ids", inputs.combo_position_ids);
    addTensor(payload, "last_hidden_states", inputs.last_hidden_states);
    addTensor(payload, "attention_mask", inputs.attention_mask);
    addTensor(payload, "kv_cache_block_id", inputs.kv_cache_block_id);
    addTensor(payload, "kv_cache_layer_to_group", inputs.kv_cache_layer_to_group);
    addTensor(payload, "kv_cache_group_types", inputs.kv_cache_group_types);
    addTensor(payload, "kv_cache_update_mapping", inputs.kv_cache_update_mapping);
    addTensor(payload, "request_id", inputs.request_id);
    addTensor(payload, "request_pd_separation", inputs.request_pd_separation);

    addSize(payload, "kv_block_stride_bytes", inputs.kv_block_stride_bytes);
    addSize(payload, "kv_scale_stride_bytes", inputs.kv_scale_stride_bytes);
    addSize(payload, "seq_size_per_block", inputs.seq_size_per_block);
    addSize(payload, "kernel_seq_size_per_block", inputs.kernel_seq_size_per_block);
    payload.insert("pd_separation", inputs.pd_separation);
    payload.insert("decode_entrance", inputs.decode_entrance);
    payload.insert("need_all_logits", inputs.need_all_logits);
    payload.insert("need_moe_gating", inputs.need_moe_gating);
    payload.insert("warmup", inputs.warmup);
    payload.insert("skip_run", inputs.skip_run);
    payload.insert("is_fake_stream", inputs.is_fake_stream);
    payload.insert("is_target_verify", inputs.is_target_verify);

    return payload;
}

}  // namespace

ModelInputsLogger::ModelInputsLogger(int64_t rank_id,
                                     int /*backup_count*/,
                                     kmonitor::MetricsReporterPtr metrics_reporter):
    rank_id_(rank_id), metrics_reporter_(std::move(metrics_reporter)) {}

ModelInputsLogger::~ModelInputsLogger() = default;

void ModelInputsLogger::initOutputDir() {
    const auto log_path = autil::EnvUtil::getEnv("LOG_PATH", std::string("logs"));
    server_id_          = autil::EnvUtil::getEnv("FRONTEND_SERVER_ID", 0);
    output_dir_         = (std::filesystem::path(log_path)
                   / ("model_inputs_r" + std::to_string(rank_id_) + "_s" + std::to_string(server_id_)))
                      .string();

    const auto      log_dir = std::filesystem::path(output_dir_);
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    if (ec) {
        RTP_LLM_LOG_WARNING(
            "Failed to create model inputs dump directory %s: %s", output_dir_.c_str(), ec.message().c_str());
        output_dir_valid_ = false;
        return;
    }
    output_dir_valid_ = true;
}

void ModelInputsLogger::log(const GptModelInputs& inputs) {
    if (inputs.is_fake_stream) {
        return;
    }

    const auto total_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    std::call_once(init_once_, [&]() { initOutputDir(); });
    if (!output_dir_valid_) {
        return;
    }

    try {
        const auto now_us     = autil::TimeUtility::currentTimeInMicroSeconds();
        const auto dump_index = dump_index_.fetch_add(1, std::memory_order_relaxed);
        const auto file_path  = std::filesystem::path(output_dir_)
                               / ("model_inputs_r" + std::to_string(rank_id_) + "_s" + std::to_string(server_id_) + "_"
                                  + timestampForFile(now_us) + "_" + std::to_string(dump_index) + ".pt");
        const auto tmp_path = file_path.string() + ".tmp";

        auto          payload = buildModelInputsPayload(inputs);
        auto          pickled = torch::pickle_save(c10::IValue(std::move(payload)));
        std::ofstream output(tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            RTP_LLM_LOG_WARNING("Failed to open model inputs dump file %s", tmp_path.c_str());
            return;
        }
        output.write(pickled.data(), pickled.size());
        output.close();

        std::error_code ec;
        std::filesystem::rename(tmp_path, file_path, ec);
        if (ec) {
            RTP_LLM_LOG_WARNING("Failed to rename model inputs dump file from %s to %s: %s",
                                tmp_path.c_str(),
                                file_path.c_str(),
                                ec.message().c_str());
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to dump model inputs: %s", e.what());
    }

    const auto total_us = autil::TimeUtility::currentTimeInMicroSeconds() - total_start_us;
    if (metrics_reporter_) {
        metrics_reporter_->report(total_us, "rtp_llm_model_inputs_log_us", kmonitor::MetricType::GAUGE, nullptr, true);
    }
}

}  // namespace rtp_llm

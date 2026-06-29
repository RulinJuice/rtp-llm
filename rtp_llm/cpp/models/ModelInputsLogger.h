#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

class ModelInputsLogger {
public:
    ModelInputsLogger(int64_t rank_id, int backup_count, kmonitor::MetricsReporterPtr metrics_reporter);
    ~ModelInputsLogger();

    void log(const GptModelInputs& inputs);

private:
    void initOutputDir();

    int64_t                      rank_id_{0};
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::once_flag               init_once_;
    std::string                  output_dir_;
    int                          server_id_{0};
    std::atomic<uint64_t>        dump_index_{0};
    bool                         output_dir_valid_{false};
};

}  // namespace rtp_llm

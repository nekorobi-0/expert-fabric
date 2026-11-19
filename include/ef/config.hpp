#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ef {

struct EngineConfig {
    std::string id = "default";
    std::size_t capacity_bytes = 64 * 1024 * 1024;
    std::string transport = "local";
    std::string rdma_endpoint;
    std::string rdma_source_address;
    bool cpu_vectorize = false;
    std::optional<unsigned> gpu_device;
    std::optional<std::size_t> bandwidth_bytes_per_second;
    std::optional<unsigned> worker_numa_node;
};

struct NgramConfig {
    std::size_t min_match = 2;
    std::size_t max_match = 5;
    std::size_t max_draft = 4;
    std::size_t max_bytes = 64 * 1024 * 1024;
};

struct Config {
    std::string metrics_listen = "127.0.0.1:9464";
    bool metrics_enabled = false;
    bool per_expert_counters = false;
    std::optional<unsigned> main_numa_node;
    std::optional<NgramConfig> ngram;
    std::vector<EngineConfig> engines{EngineConfig{}};
};

Config load_config(const std::filesystem::path& path);

}  // namespace ef

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace ef {

class Metrics {
    struct ExpertCounters;

public:
    class ExpertHandle {
    public:
        ExpertHandle() = default;
        void record(std::size_t tokens) const noexcept;

    private:
        friend class Metrics;
        explicit ExpertHandle(ExpertCounters& counters) noexcept : counters_(&counters) {}
        ExpertCounters* counters_ = nullptr;
    };

    explicit Metrics(bool per_expert = false);

    void record_stage(std::string_view stage, double seconds);
    void record_request(double seconds);
    void record_token_gap(double seconds);
    void record_tokens(std::string_view kind, std::size_t count);
    void record_invocation(std::string_view expert, std::size_t tokens);
    // Cache the handle for repeated observations. It is valid while this registry exists.
    [[nodiscard]] ExpertHandle expert_counter(std::string_view expert);
    [[nodiscard]] std::string exposition() const;

private:
    struct Histogram {
        void observe(double seconds);
        void write(std::ostream& output, std::string_view name, std::string_view labels) const;

        std::array<std::uint64_t, 14> bins{};
        std::uint64_t count = 0;
        double total = 0.0;
    };

    struct ExpertCounters {
        std::atomic<std::uint64_t> invocations{0};
        std::atomic<std::uint64_t> tokens{0};
    };

    const bool per_expert_;
    mutable std::mutex mutex_;
    std::map<std::string, Histogram, std::less<>> stages_;
    std::map<std::string, ExpertCounters, std::less<>> experts_;
    std::map<std::string, std::uint64_t, std::less<>> tokens_;
    Histogram requests_;
    Histogram token_gaps_;
};

class MetricsServer {
public:
    // The metrics registry must outlive the server.
    MetricsServer(std::string_view listen, const Metrics& metrics);
    ~MetricsServer();

    MetricsServer(const MetricsServer&) = delete;
    MetricsServer& operator=(const MetricsServer&) = delete;

    [[nodiscard]] std::string address() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ef

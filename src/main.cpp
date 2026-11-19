#include "ef/config.hpp"
#include "ef/expert.hpp"
#include "ef/metrics.hpp"
#include "ef/ngram.hpp"
#include "ef/roce.hpp"

#include <linux/mempolicy.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

template <typename Number>
Number number(std::string_view value, std::string_view field) {
    Number result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error("INVALID_ARGUMENT: invalid " + std::string(field));
    }
    return result;
}

struct Affinity {
    cpu_set_t cpus{};
    std::set<unsigned> sockets;
};

Affinity affinity_for(unsigned node, const cpu_set_t& allowed) {
    const auto directory = std::filesystem::path("/sys/devices/system/node") /
                           ("node" + std::to_string(node));
    std::ifstream cpu_list(directory / "cpulist");
    std::string list;
    if (!(cpu_list >> list)) {
        throw std::runtime_error("NUMA node " + std::to_string(node) + " is unavailable");
    }

    Affinity result;
    std::istringstream ranges(list);
    for (std::string range; std::getline(ranges, range, ',');) {
        const auto separator = range.find('-');
        const auto first = number<unsigned>(range.substr(0, separator), "CPU number");
        const auto last = separator == range.npos
                              ? first
                              : number<unsigned>(range.substr(separator + 1), "CPU number");
        for (auto cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) {
                continue;
            }
            const auto path = std::filesystem::path("/sys/devices/system/cpu") /
                              ("cpu" + std::to_string(cpu)) / "topology/physical_package_id";
            std::ifstream file(path);
            unsigned socket{};
            if (!(file >> socket)) {
                throw std::runtime_error("physical socket topology is unavailable");
            }
            CPU_SET(cpu, &result.cpus);
            result.sockets.insert(socket);
        }
    }

    if (result.sockets.empty()) {
        throw std::runtime_error("NUMA node " + std::to_string(node) + " has no permitted CPU");
    }
    return result;
}

void bind_cpu(const Affinity& affinity) {
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &affinity.cpus);
    if (error) {
        throw std::system_error(error, std::generic_category(), "setting CPU affinity");
    }
}

void bind_memory(unsigned node) {
    constexpr auto bits = std::numeric_limits<unsigned long>::digits;
    std::vector<unsigned long> mask(node / bits + 1);
    mask[node / bits] = 1UL << (node % bits);
    // Linux derives the mask width from maxnode minus one.
    if (::syscall(SYS_set_mempolicy, MPOL_BIND, mask.data(), mask.size() * bits + 1) != 0) {
        throw std::system_error(errno, std::generic_category(), "setting NUMA memory policy");
    }
}

bool upi_visible() {
    std::error_code error;
    for (std::filesystem::directory_iterator source("/sys/bus/event_source/devices", error), end;
         !error && source != end; source.increment(error)) {
        if (source->path().filename().string().starts_with("uncore_upi_")) {
            return true;
        }
    }
    return false;
}

class LocalWorker {
public:
    LocalWorker(const ef::EngineConfig& config, std::optional<unsigned> main_node) {
        cpu_set_t allowed{};
        const int error = pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed);
        if (error) {
            throw std::system_error(error, std::generic_category(), "getting CPU affinity");
        }
        const auto main = main_node
                              ? std::optional(affinity_for(*main_node, allowed))
                              : std::nullopt;
        const auto worker = config.worker_numa_node
                                ? std::optional(affinity_for(*config.worker_numa_node, allowed))
                                : std::nullopt;
        if (config.transport == "upi" &&
            (!main || !worker || main->sockets.size() != 1 || worker->sockets.size() != 1 ||
             main->sockets == worker->sockets || !upi_visible())) {
            throw std::runtime_error("UNAVAILABLE_TRANSPORT: UPI requires two accessible Intel UPI sockets and both NUMA node IDs");
        }

        std::promise<void> startup;
        auto started = startup.get_future();
        thread_ = std::thread([this, worker, node = config.worker_numa_node,
                               upi = config.transport == "upi", signal = std::move(startup)]() mutable {
            try {
                if (worker) {
                    bind_cpu(*worker);
                }
                if (upi) {
                    bind_memory(*node);
                }
                signal.set_value();
            } catch (...) {
                signal.set_exception(std::current_exception());
                return;
            }
            run();
        });
        try {
            started.get();
        } catch (...) {
            stop();
            throw;
        }
    }

    ~LocalWorker() { stop(); }

    template <typename Job>
    std::string execute(Job job) {
        std::packaged_task<std::string()> task(std::move(job));
        auto result = task.get_future();
        {
            std::lock_guard lock(mutex_);
            jobs_.push(std::move(task));
        }
        ready_.notify_one();
        return result.get();
    }

private:
    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void run() {
        for (;;) {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
            if (jobs_.empty()) {
                return;
            }
            auto job = std::move(jobs_.front());
            jobs_.pop();
            lock.unlock();
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::packaged_task<std::string()>> jobs_;
    bool stopping_ = false;
    std::thread thread_;
};

std::string argument(std::istream& line, std::string_view field) {
    std::string value;
    if (!(line >> value)) {
        throw std::runtime_error("INVALID_ARGUMENT: missing " + std::string(field));
    }
    return value;
}

void end_of_line(std::istream& line) {
    std::string extra;
    if (line >> extra) {
        throw std::runtime_error("INVALID_ARGUMENT: unexpected argument");
    }
}

using ExpertCounters = std::map<std::string, ef::Metrics::ExpertHandle, std::less<>>;

std::string ngram_command(std::string_view operation, std::string line, ef::NgramCache* cache,
                          ef::Metrics* metrics) {
    if (!cache) return "ERR UNSUPPORTED_FEATURE: n-gram cache is disabled";
    try {
        std::istringstream input(std::move(line));
        const auto model = argument(input, "model");
        const auto tokenizer = argument(input, "tokenizer");
        const auto scope = argument(input, "sharing scope");
        const ef::NgramPartition partition{model, tokenizer, scope};
        std::vector<ef::NgramCache::Token> tokens;
        for (std::string value; input >> value;) {
            tokens.push_back(number<ef::NgramCache::Token>(value, "token ID"));
        }
        const auto start = Clock::now();
        if (operation == "NGRAM_OBSERVE") {
            cache->observe(partition, tokens);
            return "OK";
        }
        const auto candidates = cache->propose(partition, tokens);
        if (metrics) {
            metrics->record_stage("ngram_lookup", std::chrono::duration<double>(Clock::now() - start).count());
            metrics->record_tokens("ngram_proposed", candidates.size());
        }
        std::string response = "OK";
        for (const auto token : candidates) response += ' ' + std::to_string(token);
        return response;
    } catch (const std::invalid_argument& error) {
        return std::string("ERR INVALID_ARGUMENT: ") + error.what();
    } catch (const std::exception& error) {
        return std::string("ERR ") + error.what();
    }
}

std::string command(std::string line, ef::ExpertEngine& engine, ef::Metrics* metrics,
                    ExpertCounters* counters, Clock::time_point queued,
                    const std::optional<std::filesystem::path>& weights_root,
                    std::optional<std::size_t> bandwidth) {
    if (line.find_first_of("\r\n\0", 0, 3) != line.npos) {
        throw std::runtime_error("INVALID_ARGUMENT: expected a single command line");
    }
    const auto started = metrics ? Clock::now() : Clock::time_point{};
    std::istringstream input(std::move(line));
    const auto operation = argument(input, "operation");
    if (operation == "HELLO") {
        end_of_line(input);
        std::string result = engine.gpu_enabled() ? "OK GPU" : "OK CPU";
        result += " max_memory_bytes=" + std::to_string(engine.capacity_bytes()) +
                  " available_memory_bytes=" + std::to_string(engine.available_bytes());
        if (bandwidth) result += " bandwidth_bytes_per_second=" + std::to_string(*bandwidth);
        return result;
    }

    if (operation != "STATUS" && operation != "LOAD" && operation != "PUBLISH" && operation != "INVOKE" &&
        operation != "RETIRE") {
        throw std::runtime_error("INVALID_ARGUMENT: unknown operation");
    }
    const auto expert = argument(input, "expert");
    const auto generation = number<std::uint64_t>(argument(input, "generation"), "generation");
    if (operation == "STATUS") {
        end_of_line(input);
        return "OK " + engine.status(expert, generation);
    }
    if (operation == "LOAD") {
        std::string file;
        if (!(input >> std::quoted(file)) || file.empty()) {
            throw std::runtime_error("INVALID_ARGUMENT: expected weights file");
        }
        end_of_line(input);
        std::filesystem::path weights(file);
        if (weights_root) {
            std::error_code error;
            weights = std::filesystem::canonical(*weights_root / weights, error);
            const auto relative = weights.lexically_relative(*weights_root);
            if (error || relative.empty() || *relative.begin() == ".." ||
                !std::filesystem::is_regular_file(weights, error)) {
                throw std::runtime_error("LOAD_FAILED: weights must be a regular file within the worker weights directory");
            }
        }
        const auto start = metrics ? Clock::now() : Clock::time_point{};
        engine.load(expert, generation, weights);
        if (metrics) {
            metrics->record_stage("weight_load", std::chrono::duration<double>(Clock::now() - start).count());
        }
        if (counters && !counters->contains(expert)) {
            counters->emplace(expert, metrics->expert_counter(expert));
        }
        return "OK READY";
    }
    if (operation == "PUBLISH") {
        const auto previous = argument(input, "expected generation (or -)");
        end_of_line(input);
        const auto expected = previous == "-" ? std::nullopt
                                               : std::optional(number<std::uint64_t>(previous, "expected generation"));
        engine.publish(expert, generation, expected);
        return "OK ACTIVE";
    }
    if (operation == "RETIRE") {
        end_of_line(input);
        engine.retire(expert, generation);
        return "OK";
    }

    std::vector<float> hidden;
    for (std::string token; input >> token;) {
        const auto value = number<float>(token, "input value");
        if (!std::isfinite(value)) {
            throw std::runtime_error("INVALID_ARGUMENT: input must be finite");
        }
        hidden.push_back(value);
    }
    ef::InvocationStats stats{};
    const auto output = engine.invoke(expert, generation, hidden, metrics ? &stats : nullptr);
    if (metrics) {
        const double queue_duration = std::chrono::duration<double>(started - queued).count();
        metrics->record_stage("expert_queue", queue_duration + stats.queue_seconds);
        metrics->record_stage("expert_compute", stats.compute_seconds);
    }
    if (counters) {
        const auto counter = counters->find(expert);
        if (counter != counters->end()) {
            counter->second.record(stats.tokens);
        }
    }

    std::ostringstream response;
    response << "OK" << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto value : output) {
        response << ' ' << value;
    }
    return response.str();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: expert-fabric [--config PATH]\n"
                         "       expert-fabric [--config PATH] --serve-rdma ADDRESS --weights-root DIR\n"
                         "Commands: HELLO, LOAD expert generation path, PUBLISH expert generation expected-or-dash,\n"
                         "          INVOKE expert generation float..., STATUS expert generation, RETIRE expert generation, QUIT\n"
                         "          ENGINES, AUTOLOAD expert generation path, @engine COMMAND\n"
                         "          NGRAM_OBSERVE / NGRAM_PROPOSE model tokenizer scope token...\n";
            return 0;
        }
        ef::Config config;
        std::optional<std::filesystem::path> config_path;
        std::optional<std::filesystem::path> weights_root;
        std::optional<std::string> listen;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (i + 1 < argc && option == "--config" && !config_path) {
                config_path = argv[++i];
            } else if (i + 1 < argc && option == "--weights-root" && !weights_root) {
                weights_root = argv[++i];
            } else if (i + 1 < argc && option == "--serve-rdma" && !listen) {
                listen = argv[++i];
            } else {
                throw std::runtime_error("invalid options; see expert-fabric --help");
            }
        }
        if (config_path) {
            config = ef::load_config(*config_path);
        }
        if (listen.has_value() != weights_root.has_value()) {
            throw std::runtime_error("--serve-rdma and --weights-root must be used together");
        }
        if (weights_root) {
            *weights_root = std::filesystem::canonical(*weights_root);
            if (!std::filesystem::is_directory(*weights_root)) {
                throw std::runtime_error("--weights-root must be a directory");
            }
        }

        if (listen && (config.engines.size() != 1 || config.engines.front().transport == "roce_v2")) {
            throw std::runtime_error("the RoCE worker needs one local or UPI expert engine");
        }
        struct Target {
            std::unique_ptr<ef::ExpertEngine> engine;
            std::unique_ptr<LocalWorker> worker;
            std::unique_ptr<ef::RoceClient> client;
        };
        std::vector<Target> targets;
        targets.reserve(config.engines.size());
        for (const auto& settings : config.engines) {
            auto& target = targets.emplace_back();
            if (settings.transport != "roce_v2") {
                target.engine = std::make_unique<ef::ExpertEngine>(
                    settings.capacity_bytes, settings.cpu_vectorize, settings.gpu_device);
                target.worker = std::make_unique<LocalWorker>(settings, config.main_numa_node);
            }
        }
        if (config.main_numa_node) {
            cpu_set_t allowed{};
            const int error = pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed);
            if (error) throw std::system_error(error, std::generic_category(), "getting main CPU affinity");
            bind_cpu(affinity_for(*config.main_numa_node, allowed));
        }
        for (std::size_t index = 0; index < targets.size(); ++index) {
            const auto& settings = config.engines[index];
            if (settings.transport == "roce_v2") {
                targets[index].client = std::make_unique<ef::RoceClient>(
                    settings.rdma_endpoint, settings.rdma_source_address);
            }
        }

        ef::Metrics metrics(config.per_expert_counters);
        ExpertCounters counters;
        std::unique_ptr<ef::MetricsServer> exporter;
        if (config.metrics_enabled) {
            exporter = std::make_unique<ef::MetricsServer>(config.metrics_listen, metrics);
            std::cerr << "Metrics: http://" << exporter->address() << "/metrics\n" << std::flush;
        }

        const auto execute = [&](std::size_t index, std::string line) {
            const auto start = exporter ? Clock::now() : Clock::time_point{};
            auto& target = targets[index];
            try {
                std::string response;
                if (target.client) {
                    response = target.client->execute(line);
                } else {
                    response = target.worker->execute([&, line = std::move(line), start]() mutable {
                        return command(std::move(line), *target.engine, exporter ? &metrics : nullptr,
                                       exporter && config.per_expert_counters ? &counters : nullptr,
                                       start, weights_root, config.engines[index].bandwidth_bytes_per_second);
                    });
                }
                if (exporter) {
                    metrics.record_stage("command_roundtrip", std::chrono::duration<double>(Clock::now() - start).count());
                }
                return response;
            } catch (const std::exception& error) {
                return std::string("ERR ") + error.what();
            }
        };
        if (listen) {
            ef::run_roce_server(*listen, [&](std::string line) { return execute(0, std::move(line)); });
            return 0;
        }

        std::unique_ptr<ef::NgramCache> ngram;
        if (const auto& settings = config.ngram) {
            ngram = std::make_unique<ef::NgramCache>(
                settings->min_match, settings->max_match, settings->max_draft, settings->max_bytes);
        }
        const auto dispatch = [&](std::string line) {
            if (line == "ENGINES") {
                std::string report = "OK";
                for (std::size_t index = 0; index < targets.size(); ++index) {
                    const auto& settings = config.engines[index];
                    report += " | " + settings.id + " " + settings.transport + " " + execute(index, "HELLO");
                }
                return report;
            }
            const auto operation = line.substr(0, line.find_first_of(" \t"));
            if (operation == "NGRAM_OBSERVE" || operation == "NGRAM_PROPOSE") {
                return ngram_command(operation, line.substr(operation.size()), ngram.get(), exporter ? &metrics : nullptr);
            }
            if (operation == "AUTOLOAD") {
                const auto request = "LOAD" + line.substr(operation.size());
                std::vector<std::pair<std::size_t, std::size_t>> placement;
                for (std::size_t index = 0; index < targets.size(); ++index) {
                    const auto configured = config.engines[index].bandwidth_bytes_per_second;
                    std::size_t bandwidth = configured.value_or(0);
                    if (targets[index].client && !configured) {
                        const auto report = execute(index, "HELLO");
                        constexpr std::string_view key = " bandwidth_bytes_per_second=";
                        if (const auto field = report.find(key); report.starts_with("OK ") && field != report.npos) {
                            const auto end = report.data() + report.size();
                            const auto parsed = std::from_chars(report.data() + field + key.size(), end, bandwidth);
                            if (parsed.ec != std::errc{} || parsed.ptr != end) bandwidth = 0;
                        }
                    }
                    placement.emplace_back(bandwidth, index);
                }
                std::ranges::stable_sort(placement, std::greater{}, [](const auto& candidate) { return candidate.first; });
                for (const auto& candidate : placement) {
                    const auto index = candidate.second;
                    auto response = execute(index, request);
                    if (response == "OK READY") return "OK READY @" + config.engines[index].id;
                    if (!response.starts_with("ERR NO_CAPACITY:")) return response;
                }
                return std::string("ERR NO_CAPACITY: no engine has room for weights");
            }
            if (operation.starts_with('@')) {
                const auto target = std::ranges::find(config.engines, operation.substr(1), &ef::EngineConfig::id);
                if (target == config.engines.end()) return std::string("ERR INVALID_ARGUMENT: unknown engine");
                return execute(static_cast<std::size_t>(target - config.engines.begin()), line.substr(operation.size()));
            }
            return execute(0, std::move(line));
        };

        for (std::string line; std::getline(std::cin, line);) {
            if (line == "QUIT" || line == "QUIT\r") {
                break;
            }
            if (line.ends_with('\r')) {
                line.pop_back();
            }
            if (line.find_first_not_of(" \t\r") == line.npos) {
                continue;
            }
            std::cout << dispatch(std::move(line)) << '\n' << std::flush;
        }
    } catch (const std::exception& error) {
        std::cerr << "expert-fabric: " << error.what() << '\n';
        return 1;
    }
}

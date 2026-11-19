#include "ef/metrics.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace ef {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct Bucket {
    double bound;
    std::string_view text;
};

constexpr std::array buckets{
    Bucket{0.000001, "0.000001"}, Bucket{0.000005, "0.000005"},
    Bucket{0.00001, "0.00001"}, Bucket{0.000025, "0.000025"},
    Bucket{0.00005, "0.00005"}, Bucket{0.0001, "0.0001"},
    Bucket{0.00025, "0.00025"}, Bucket{0.0005, "0.0005"},
    Bucket{0.001, "0.001"}, Bucket{0.005, "0.005"},
    Bucket{0.01, "0.01"}, Bucket{0.1, "0.1"}, Bucket{1.0, "1"},
    Bucket{10.0, "10"},
};

bool valid_duration(double seconds) {
    return std::isfinite(seconds) && seconds >= 0;
}

template <typename Map>
decltype(auto) find_or_add(Map& map, std::string_view key) {
    const auto existing = map.find(key);
    if (existing != map.end()) {
        return (existing->second);
    }
    return (map.try_emplace(std::string(key)).first->second);
}

std::string escape_label(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char byte : value) {
        switch (byte) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        default: result += byte; break;
        }
    }
    return result;
}

void write_header(std::ostream& output, std::string_view name, std::string_view type,
                  std::string_view description) {
    output << "# HELP " << name << ' ' << description << '\n';
    output << "# TYPE " << name << ' ' << type << '\n';
}

class Socket {
public:
    explicit Socket(int fd) : fd_(fd) {}
    ~Socket() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    Socket(Socket&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] int get() const { return fd_; }

private:
    int fd_;
};

std::pair<std::string, std::string> split_address(std::string_view listen) {
    const auto separator = listen.rfind(':');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 == listen.size()) {
        throw std::invalid_argument("metrics listen address must be host:port or [IPv6]:port");
    }
    std::string_view host = listen.substr(0, separator);
    if (host.starts_with('[') && host.ends_with(']')) {
        host.remove_prefix(1);
        host.remove_suffix(1);
    } else if (host.find(':') != std::string_view::npos) {
        throw std::invalid_argument("metrics IPv6 listen address must use brackets");
    }

    const std::string_view port = listen.substr(separator + 1);
    unsigned value = 0;
    const auto [end, error] = std::from_chars(port.data(), port.data() + port.size(), value);
    if (host.empty() || host.find('\0') != std::string_view::npos || error != std::errc{} ||
        end != port.data() + port.size() || value > 65535) {
        throw std::invalid_argument("invalid metrics host or port");
    }
    return {std::string(host), std::to_string(value)};
}

Socket open_listener(std::string_view listen) {
    const auto [host, port] = split_address(listen);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICSERV;

    addrinfo* addresses = nullptr;
    const int status = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses);
    if (status != 0) {
        throw std::runtime_error(std::string("resolve metrics listen address: ") + ::gai_strerror(status));
    }
    const std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> results(addresses, ::freeaddrinfo);
    int last_error = EADDRNOTAVAIL;
    for (const auto* entry = results.get(); entry; entry = entry->ai_next) {
        Socket socket(::socket(entry->ai_family, entry->ai_socktype | SOCK_CLOEXEC | SOCK_NONBLOCK,
                               entry->ai_protocol));
        if (socket.get() >= 0) {
            const int reuse = 1;
            if (::setsockopt(socket.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == 0 &&
                ::bind(socket.get(), entry->ai_addr, entry->ai_addrlen) == 0 &&
                ::listen(socket.get(), 16) == 0) {
                return socket;
            }
        }
        last_error = errno;
    }
    throw std::system_error(last_error, std::generic_category(), "bind metrics listen address");
}

std::string describe_listener(int fd) {
    sockaddr_storage address{};
    auto* socket_address = reinterpret_cast<sockaddr*>(&address);
    socklen_t length = sizeof(address);
    if (::getsockname(fd, socket_address, &length) != 0) {
        throw std::system_error(errno, std::generic_category(), "read metrics listen address");
    }

    std::array<char, NI_MAXHOST> host{};
    std::array<char, NI_MAXSERV> port{};
    const int status = ::getnameinfo(socket_address, length, host.data(), host.size(), port.data(),
                                     port.size(), NI_NUMERICHOST | NI_NUMERICSERV);
    if (status != 0) {
        throw std::runtime_error(std::string("format metrics listen address: ") + ::gai_strerror(status));
    }
    if (address.ss_family == AF_INET6) {
        return '[' + std::string(host.data()) + "]:" + port.data();
    }
    return std::string(host.data()) + ':' + port.data();
}

bool wait_socket(int fd, short events, std::stop_token stop, Clock::time_point deadline) {
    while (!stop.stop_requested() && Clock::now() < deadline) {
        pollfd descriptor{fd, events, 0};
        const int ready = ::poll(&descriptor, 1, 25);
        if (ready > 0) {
            return (descriptor.revents & events) != 0;
        }
        if (ready < 0 && errno != EINTR) {
            return false;
        }
    }
    return false;
}

void send_response(int fd, std::stop_token stop, std::string_view status,
                   std::string_view content_type, std::string body) {
    std::string message = "HTTP/1.1 ";
    message += status;
    message += "\r\nConnection: close\r\nContent-Type: ";
    message += content_type;
    message += "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    message += body;

    const auto deadline = Clock::now() + 500ms;
    std::string_view remaining(message);
    while (!remaining.empty() && wait_socket(fd, POLLOUT, stop, deadline)) {
        const auto written = ::send(fd, remaining.data(), remaining.size(), MSG_NOSIGNAL);
        if (written > 0) {
            remaining.remove_prefix(static_cast<std::size_t>(written));
        } else if (written == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
            return;
        }
    }
}

void serve_client(int fd, const Metrics& metrics, std::stop_token stop) {
    constexpr std::size_t max_headers_bytes = 8192;
    std::array<char, 1024> buffer{};
    std::string request;
    const auto deadline = Clock::now() + 500ms;
    while (wait_socket(fd, POLLIN, stop, deadline)) {
        const auto bytes = ::recv(fd, buffer.data(), buffer.size(), 0);
        if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            continue;
        }
        if (bytes <= 0) {
            return;
        }
        request.append(buffer.data(), static_cast<std::size_t>(bytes));
        if (request.size() > max_headers_bytes) {
            send_response(fd, stop, "431 Request Header Fields Too Large", "text/plain", "headers too large\n");
            return;
        }
        if (request.find("\r\n\r\n") == std::string::npos) {
            continue;
        }

        const auto first_line = std::string_view(request).substr(0, request.find("\r\n"));
        const auto method_end = first_line.find(' ');
        const auto target_end = first_line.rfind(' ');
        if (method_end == std::string_view::npos || target_end <= method_end ||
            first_line.substr(target_end + 1, 5) != "HTTP/") {
            send_response(fd, stop, "400 Bad Request", "text/plain", "bad request\n");
        } else if (first_line.substr(0, method_end) != "GET") {
            send_response(fd, stop, "405 Method Not Allowed", "text/plain", "method not allowed\n");
        } else if (first_line.substr(method_end + 1, target_end - method_end - 1) != "/metrics") {
            send_response(fd, stop, "404 Not Found", "text/plain", "not found\n");
        } else {
            send_response(fd, stop, "200 OK", "text/plain; version=0.0.4; charset=utf-8", metrics.exposition());
        }
        return;
    }
}

void serve(int listener, const Metrics& metrics, std::stop_token stop) {
    while (wait_socket(listener, POLLIN, stop, Clock::time_point::max())) {
        Socket client(::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK));
        if (client.get() < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED) {
                continue;
            }
            return;
        }
        try {
            serve_client(client.get(), metrics, stop);
        } catch (const std::exception&) {
            // Losing a scrape must not terminate the inference process.
        }
    }
}

}  // namespace

Metrics::Metrics(bool per_expert) : per_expert_(per_expert) {}

void Metrics::Histogram::observe(double seconds) {
    static_assert(std::tuple_size_v<decltype(bins)> == buckets.size());
    const auto bucket = std::lower_bound(buckets.begin(), buckets.end(), seconds,
                                         [](const Bucket& entry, double value) { return entry.bound < value; });
    if (bucket != buckets.end()) {
        ++bins[static_cast<std::size_t>(bucket - buckets.begin())];
    }
    ++count;
    total += seconds;
}

void Metrics::Histogram::write(std::ostream& output, std::string_view name, std::string_view labels) const {
    const std::string_view separator = labels.empty() ? "" : ",";
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        cumulative += bins[i];
        output << name << "_bucket{" << labels << separator << "le=\"" << buckets[i].text
               << "\"} " << cumulative << '\n';
    }
    output << name << "_bucket{" << labels << separator << "le=\"+Inf\"} " << count << '\n';
    const std::string suffix = labels.empty() ? "" : '{' + std::string(labels) + '}';
    output << name << "_sum" << suffix << ' ' << total << '\n';
    output << name << "_count" << suffix << ' ' << count << '\n';
}

void Metrics::record_stage(std::string_view stage, double seconds) {
    if (valid_duration(seconds)) {
        const std::lock_guard lock(mutex_);
        find_or_add(stages_, stage).observe(seconds);
    }
}

void Metrics::record_request(double seconds) {
    if (valid_duration(seconds)) {
        const std::lock_guard lock(mutex_);
        requests_.observe(seconds);
    }
}

void Metrics::record_token_gap(double seconds) {
    if (valid_duration(seconds)) {
        const std::lock_guard lock(mutex_);
        token_gaps_.observe(seconds);
    }
}

void Metrics::record_tokens(std::string_view kind, std::size_t count) {
    const std::lock_guard lock(mutex_);
    find_or_add(tokens_, kind) += count;
}

void Metrics::record_invocation(std::string_view expert, std::size_t tokens) {
    expert_counter(expert).record(tokens);
}

Metrics::ExpertHandle Metrics::expert_counter(std::string_view expert) {
    if (!per_expert_) {
        return {};
    }
    const std::lock_guard lock(mutex_);
    return ExpertHandle(find_or_add(experts_, expert));
}

void Metrics::ExpertHandle::record(std::size_t tokens) const noexcept {
    if (counters_ != nullptr) {
        counters_->tokens.fetch_add(tokens, std::memory_order_relaxed);
        counters_->invocations.fetch_add(1, std::memory_order_relaxed);
    }
}

std::string Metrics::exposition() const {
    struct ExpertSnapshot {
        std::string name;
        std::uint64_t invocations;
        std::uint64_t tokens;
    };
    decltype(stages_) stages;
    std::vector<ExpertSnapshot> experts;
    decltype(tokens_) tokens;
    Histogram requests;
    Histogram token_gaps;
    {
        const std::lock_guard lock(mutex_);
        stages = stages_;
        experts.reserve(experts_.size());
        for (const auto& [name, counters] : experts_) {
            experts.push_back({name, counters.invocations.load(std::memory_order_relaxed),
                               counters.tokens.load(std::memory_order_relaxed)});
        }
        tokens = tokens_;
        requests = requests_;
        token_gaps = token_gaps_;
    }

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);

    constexpr std::string_view stage_duration = "expert_fabric_stage_duration_seconds";
    write_header(output, stage_duration, "histogram", "Duration of a processing stage in seconds.");
    for (const auto& [stage, histogram] : stages) {
        histogram.write(output, stage_duration, "stage=\"" + escape_label(stage) + '"');
    }

    constexpr std::string_view request_duration = "expert_fabric_request_duration_seconds";
    write_header(output, request_duration, "histogram", "End-to-end request duration in seconds.");
    requests.write(output, request_duration, "");

    constexpr std::string_view token_gap = "expert_fabric_inter_token_gap_seconds";
    write_header(output, token_gap, "histogram", "Interval between emitted tokens in seconds.");
    token_gaps.write(output, token_gap, "");

    constexpr std::string_view token_total = "expert_fabric_tokens_total";
    write_header(output, token_total, "counter", "Number of tokens observed by kind.");
    for (const auto& [kind, count] : tokens) {
        output << token_total << "{kind=\"" << escape_label(kind) << "\"} " << count << '\n';
    }

    if (per_expert_) {
        constexpr std::string_view invocations = "expert_fabric_expert_invocations_total";
        constexpr std::string_view expert_tokens = "expert_fabric_expert_tokens_total";
        write_header(output, invocations, "counter", "Number of expert invocations.");
        for (const auto& expert : experts) {
            output << invocations << "{expert_id=\"" << escape_label(expert.name) << "\"} "
                   << expert.invocations << '\n';
        }
        write_header(output, expert_tokens, "counter", "Number of tokens passed to each expert.");
        for (const auto& expert : experts) {
            output << expert_tokens << "{expert_id=\"" << escape_label(expert.name) << "\"} "
                   << expert.tokens << '\n';
        }
    }
    return std::move(output).str();
}

struct MetricsServer::Impl {
    Impl(std::string_view listen, const Metrics& metrics)
        : listener(open_listener(listen)), bound_address(describe_listener(listener.get())),
          worker([fd = listener.get(), &metrics](std::stop_token stop) { serve(fd, metrics, stop); }) {}

    Socket listener;
    std::string bound_address;
    // Destroy the worker before closing the listening socket.
    std::jthread worker;
};

MetricsServer::MetricsServer(std::string_view listen, const Metrics& metrics)
    : impl_(std::make_unique<Impl>(listen, metrics)) {}

MetricsServer::~MetricsServer() = default;

std::string MetricsServer::address() const { return impl_->bound_address; }

}  // namespace ef

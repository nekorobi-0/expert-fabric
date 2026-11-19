#include "ef/roce.hpp"

#include <stdexcept>

#ifdef EF_HAVE_RDMA

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <rdma/rdma_cma.h>

namespace ef {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t message_limit = 1024 * 1024;
constexpr auto setup_timeout = std::chrono::seconds(5);
constexpr auto request_timeout = std::chrono::minutes(5);
constexpr std::array<unsigned char, 4> protocol_version{'E', 'F', 1, 0};
constexpr std::uint64_t receive_work = 1;
constexpr std::uint64_t send_work = 2;

[[noreturn]] void raise_errno(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

void check_status(int status, const char* operation) {
    if (status != 0) {
        throw std::system_error(status < 0 ? errno : status, std::generic_category(), operation);
    }
}

template <typename T, auto Destroy>
struct Deleter {
    void operator()(T* value) const noexcept { Destroy(value); }
};

template <typename T, auto Destroy>
using Owner = std::unique_ptr<T, Deleter<T, Destroy>>;

template <typename T>
T* check_pointer(T* value, const char* operation) {
    if (!value) {
        raise_errno(operation);
    }
    return value;
}

using Channel = Owner<rdma_event_channel, rdma_destroy_event_channel>;
using Identifier = Owner<rdma_cm_id, rdma_destroy_id>;
using Event = Owner<rdma_cm_event, rdma_ack_cm_event>;
using CompletionChannel = Owner<ibv_comp_channel, ibv_destroy_comp_channel>;
using CompletionQueue = Owner<ibv_cq, ibv_destroy_cq>;
using MemoryRegion = Owner<ibv_mr, ibv_dereg_mr>;

Channel make_channel() {
    Channel channel(check_pointer(rdma_create_event_channel(),
                                  "open RDMA CM event channel (/dev/infiniband/rdma_cm)"));
    const int flags = fcntl(channel->fd, F_GETFL);
    if (flags == -1 || fcntl(channel->fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        raise_errno("configure nonblocking RDMA CM event channel");
    }
    return channel;
}

Identifier make_id(rdma_event_channel* channel) {
    rdma_cm_id* id = nullptr;
    check_status(rdma_create_id(channel, &id, nullptr, RDMA_PS_TCP), "create RDMA CM identifier");
    return Identifier(id);
}

using Addresses = Owner<addrinfo, freeaddrinfo>;

Addresses resolve_endpoint(std::string_view endpoint, bool passive) {
    auto separator = endpoint.rfind(':');
    if (separator == endpoint.npos || separator == 0 || separator + 1 == endpoint.size()) {
        throw std::invalid_argument("RoCEv2 endpoint must be host:port or [ipv6]:port");
    }
    auto host = endpoint.substr(0, separator);
    const auto port = endpoint.substr(separator + 1);
    if (host.starts_with('[') && host.ends_with(']')) {
        host = host.substr(1, host.size() - 2);
    } else if (host.find_first_of("[]:") != host.npos) {
        throw std::invalid_argument("RoCEv2 IPv6 endpoints require brackets");
    }
    unsigned port_number = 0;
    for (const char digit : port) {
        if (digit < '0' || digit > '9') {
            throw std::invalid_argument("RoCEv2 port must be numeric");
        }
        port_number = port_number * 10 + static_cast<unsigned>(digit - '0');
        if (port_number > 65535) {
            throw std::invalid_argument("RoCEv2 port must be between 1 and 65535");
        }
    }
    if (host.empty() || port_number == 0 || (!passive && host == "*") ||
        host.find('\0') != host.npos) {
        throw std::invalid_argument("RoCEv2 endpoint requires a host and a port between 1 and 65535");
    }

    const std::string host_text(host);
    const std::string port_text(port);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_NUMERICSERV | (passive ? AI_PASSIVE : 0);
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    const int result = getaddrinfo(host == "*" ? nullptr : host_text.c_str(),
                                   port_text.c_str(), &hints, &addresses);
    if (result != 0) {
        throw std::runtime_error(std::string("resolve RoCEv2 endpoint: ") + gai_strerror(result));
    }
    return Addresses(addresses);
}

void poll_files(pollfd* files, nfds_t count, Clock::time_point deadline) {
    while (true) {
        int timeout = -1;
        if (deadline != Clock::time_point::max()) {
            const auto remaining = deadline - Clock::now();
            if (remaining <= Clock::duration::zero()) {
                throw std::runtime_error("RoCEv2 operation timed out");
            }
            const auto millis = std::chrono::ceil<std::chrono::milliseconds>(remaining).count();
            timeout = static_cast<int>(std::min<long long>(millis, std::numeric_limits<int>::max()));
        }
        const int ready = poll(files, count, timeout);
        if (ready == 0) {
            throw std::runtime_error("RoCEv2 operation timed out");
        }
        if (ready == -1) {
            if (errno == EINTR) {
                continue;
            }
            raise_errno("poll RoCEv2 event channels");
        }
        for (nfds_t index = 0; index < count; ++index) {
            if ((files[index].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                throw std::runtime_error("RoCEv2 event channel failed");
            }
        }
        return;
    }
}

Event read_event(rdma_event_channel* channel) {
    rdma_cm_event* event = nullptr;
    const int result = rdma_get_cm_event(channel, &event);
    if (result == -1 && (errno == EAGAIN || errno == EINTR)) {
        return Event();
    }
    check_status(result, "read RDMA CM event");
    return Event(event);
}

Event await_event(rdma_event_channel* channel, Clock::time_point deadline) {
    while (true) {
        pollfd file{channel->fd, POLLIN, 0};
        poll_files(&file, 1, deadline);
        if (auto event = read_event(channel)) {
            return event;
        }
    }
}

[[noreturn]] void unexpected_event(const rdma_cm_event& event) {
    std::string message = std::string("RDMA CM event ") + rdma_event_str(event.event);
    if (event.status != 0) {
        message += " (status " + std::to_string(event.status) + ")";
    }
    throw std::runtime_error(message);
}

void expect_event(rdma_event_channel* channel, rdma_cm_event_type expected) {
    const auto event = await_event(channel, Clock::now() + setup_timeout);
    if (event->event != expected || event->status != 0) {
        unexpected_event(*event);
    }
}

bool supported_protocol(const rdma_cm_event& event) {
    const auto& data = event.param.conn;
    return data.private_data && data.private_data_len >= protocol_version.size() &&
           std::memcmp(data.private_data, protocol_version.data(), protocol_version.size()) == 0;
}

void verify_roce_v2(rdma_cm_id* id) {
    ibv_port_attr port{};
    check_status(ibv_query_port(id->verbs, id->port_num, &port), "query RDMA port");
    if (port.link_layer != IBV_LINK_LAYER_ETHERNET) {
        throw std::runtime_error("the selected RDMA port does not use Ethernet");
    }
    ibv_qp_attr attributes{};
    ibv_qp_init_attr initialization{};
    check_status(ibv_query_qp(id->qp, &attributes, IBV_QP_AV, &initialization), "query connected RDMA path");
    if (!attributes.ah_attr.is_global) {
        throw std::runtime_error("the selected RDMA path has no global identifier");
    }
    ibv_gid_entry gid{};
    check_status(ibv_query_gid_ex(id->verbs, id->port_num, attributes.ah_attr.grh.sgid_index,
                                 &gid, 0), "query connected RDMA GID");
    if (gid.gid_type != IBV_GID_TYPE_ROCE_V2) {
        throw std::runtime_error("the selected Ethernet RDMA GID is not RoCEv2");
    }
}

struct QueuePair {
    rdma_cm_id* id = nullptr;
    ~QueuePair() {
        if (id) {
            rdma_destroy_qp(id);
        }
    }
};

class PeerDisconnected : public std::runtime_error {
public:
    PeerDisconnected() : std::runtime_error("RoCEv2 peer disconnected") {}
};

class Connection {
public:
    Connection(Channel channel, Identifier id)
        : channel_(std::move(channel)),
          id_(std::move(id)),
          completions_(check_pointer(ibv_create_comp_channel(id_->verbs), "create RDMA completion channel")),
          queue_(check_pointer(ibv_create_cq(id_->verbs, 8, nullptr, completions_.get(), 0),
                               "create RDMA completion queue")),
          receive_buffer_(message_limit),
          send_buffer_(message_limit) {
        check_status(ibv_req_notify_cq(queue_.get(), 0), "arm RDMA completion queue");
        ibv_qp_init_attr config{};
        config.send_cq = queue_.get();
        config.recv_cq = queue_.get();
        config.sq_sig_all = 1;
        config.qp_type = IBV_QPT_RC;
        config.cap.max_send_wr = 1;
        config.cap.max_recv_wr = 1;
        config.cap.max_send_sge = 1;
        config.cap.max_recv_sge = 1;
        check_status(rdma_create_qp(id_.get(), nullptr, &config), "create reliable-connected RDMA queue pair");
        qp_.id = id_.get();

        receive_region_.reset(check_pointer(
            ibv_reg_mr(id_->pd, receive_buffer_.data(), receive_buffer_.size(), IBV_ACCESS_LOCAL_WRITE),
            "register RDMA receive memory"));
        send_region_.reset(check_pointer(
            ibv_reg_mr(id_->pd, send_buffer_.data(), send_buffer_.size(), 0), "register RDMA send memory"));
        post_receive();
    }

    ~Connection() {
        if (connected_) {
            rdma_disconnect(id_.get());
        }
    }

    void establish(bool active) {
        rdma_conn_param parameters{};
        parameters.private_data = protocol_version.data();
        parameters.private_data_len = protocol_version.size();
        parameters.retry_count = 7;
        parameters.rnr_retry_count = 7;
        check_status(active ? rdma_connect(id_.get(), &parameters) : rdma_accept(id_.get(), &parameters),
                     active ? "connect RoCEv2 peer" : "accept RoCEv2 peer");
        const auto event = await_event(channel_.get(), Clock::now() + setup_timeout);
        if (event->event != RDMA_CM_EVENT_ESTABLISHED || event->status != 0) {
            unexpected_event(*event);
        }
        connected_ = true;
        if (active && !supported_protocol(*event)) {
            throw std::runtime_error("incompatible RoCEv2 application protocol");
        }
        verify_roce_v2(id_.get());
    }

    std::string receive(Clock::time_point deadline) {
        while (!received_) {
            collect(deadline);
        }
        std::string message(receive_buffer_.data(), *received_);
        received_.reset();
        post_receive();
        return message;
    }

    void send(std::string_view message, Clock::time_point deadline) {
        if (message.size() > message_limit) {
            throw std::length_error("RoCEv2 message exceeds the 1 MiB limit");
        }
        ibv_sge segment{};
        segment.addr = reinterpret_cast<std::uintptr_t>(send_buffer_.data());
        segment.length = static_cast<std::uint32_t>(message.size());
        segment.lkey = send_region_->lkey;
        ibv_send_wr request{};
        request.wr_id = send_work;
        request.opcode = IBV_WR_SEND;
        if (!message.empty()) {
            std::memcpy(send_buffer_.data(), message.data(), message.size());
            request.sg_list = &segment;
            request.num_sge = 1;
        }
        ibv_send_wr* bad_request = nullptr;
        send_done_ = false;
        check_status(ibv_post_send(id_->qp, &request, &bad_request), "post RoCEv2 send");
        while (!send_done_) {
            collect(deadline);
        }
    }

private:
    void post_receive() {
        ibv_sge segment{};
        segment.addr = reinterpret_cast<std::uintptr_t>(receive_buffer_.data());
        segment.length = message_limit;
        segment.lkey = receive_region_->lkey;
        ibv_recv_wr request{};
        request.wr_id = receive_work;
        request.sg_list = &segment;
        request.num_sge = 1;
        ibv_recv_wr* bad_request = nullptr;
        check_status(ibv_post_recv(id_->qp, &request, &bad_request), "post RoCEv2 receive");
    }

    void collect(Clock::time_point deadline) {
        while (true) {
            ibv_wc completion{};
            const int result = ibv_poll_cq(queue_.get(), 1, &completion);
            if (result < 0) {
                throw std::runtime_error("poll RoCEv2 work completion queue failed");
            }
            if (result == 1) {
                if (completion.status != IBV_WC_SUCCESS) {
                    if (completion.status == IBV_WC_WR_FLUSH_ERR) {
                        throw PeerDisconnected();
                    }
                    throw std::runtime_error(std::string("RoCEv2 work completion: ") +
                                             ibv_wc_status_str(completion.status));
                }
                if (completion.wr_id == receive_work && completion.opcode == IBV_WC_RECV &&
                    completion.byte_len <= receive_buffer_.size()) {
                    received_ = completion.byte_len;
                } else if (completion.wr_id == send_work && completion.opcode == IBV_WC_SEND) {
                    send_done_ = true;
                } else {
                    throw std::runtime_error("unexpected RoCEv2 work completion");
                }
                return;
            }

            pollfd files[] = {{completions_->fd, POLLIN, 0}, {channel_->fd, POLLIN, 0}};
            poll_files(files, 2, deadline);
            if ((files[1].revents & POLLIN) != 0) {
                if (const auto event = read_event(channel_.get())) {
                    if (event->event == RDMA_CM_EVENT_DISCONNECTED) {
                        connected_ = false;
                        throw PeerDisconnected();
                    }
                    unexpected_event(*event);
                }
            }
            if ((files[0].revents & POLLIN) != 0) {
                ibv_cq* notified = nullptr;
                void* context = nullptr;
                check_status(ibv_get_cq_event(completions_.get(), &notified, &context),
                             "read RDMA completion event");
                ibv_ack_cq_events(notified, 1);
                check_status(ibv_req_notify_cq(notified, 0), "rearm RDMA completion queue");
            }
        }
    }

    Channel channel_;
    Identifier id_;
    CompletionChannel completions_;
    CompletionQueue queue_;
    std::vector<char> receive_buffer_;
    std::vector<char> send_buffer_;
    MemoryRegion receive_region_;
    MemoryRegion send_region_;
    QueuePair qp_;
    std::optional<std::size_t> received_;
    bool send_done_ = false;
    bool connected_ = false;
};

Addresses resolve_source(std::string_view source) {
    if (source.empty()) return {};
    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV;
    const std::string ip(source);
    addrinfo* result = nullptr;
    const int status = getaddrinfo(ip.c_str(), "0", &hints, &result);
    if (status != 0) {
        throw std::invalid_argument("invalid RoCEv2 source address " + ip + ": " + gai_strerror(status));
    }
    return Addresses(result);
}

std::unique_ptr<Connection> connect_to(std::string_view endpoint, std::string_view source_address) {
    const auto addresses = resolve_endpoint(endpoint, false);
    const auto source = resolve_source(source_address);
    std::exception_ptr last_error;
    for (auto* address = addresses.get(); address; address = address->ai_next) {
        if (source && address->ai_family != source->ai_family) continue;
        try {
            auto channel = make_channel();
            auto id = make_id(channel.get());
            const int timeout = std::chrono::duration_cast<std::chrono::milliseconds>(setup_timeout).count();
            check_status(rdma_resolve_addr(id.get(), source ? source->ai_addr : nullptr, address->ai_addr, timeout),
                         "resolve RDMA destination");
            expect_event(channel.get(), RDMA_CM_EVENT_ADDR_RESOLVED);
            check_status(rdma_resolve_route(id.get(), timeout), "resolve RDMA route");
            expect_event(channel.get(), RDMA_CM_EVENT_ROUTE_RESOLVED);
            auto connection = std::make_unique<Connection>(std::move(channel), std::move(id));
            connection->establish(true);
            return connection;
        } catch (const std::exception&) {
            last_error = std::current_exception();
        }
    }
    if (last_error) {
        std::rethrow_exception(last_error);
    }
    if (source) throw std::invalid_argument("RoCEv2 source and destination address families do not match");
    throw std::runtime_error("RoCEv2 endpoint resolved to no address");
}

Identifier bind_listener(rdma_event_channel* channel, std::string_view endpoint) {
    const auto addresses = resolve_endpoint(endpoint, true);
    std::exception_ptr last_error;
    for (auto* address = addresses.get(); address; address = address->ai_next) {
        try {
            auto listener = make_id(channel);
            check_status(rdma_bind_addr(listener.get(), address->ai_addr), "bind RDMA CM listener");
            check_status(rdma_listen(listener.get(), 8), "listen for RoCEv2 peers");
            return listener;
        } catch (const std::exception&) {
            last_error = std::current_exception();
        }
    }
    if (last_error) {
        std::rethrow_exception(last_error);
    }
    throw std::runtime_error("RoCEv2 listener resolved to no address");
}

} // namespace

struct RoceClient::Impl {
    Impl(std::string_view endpoint, std::string_view source) : connection(connect_to(endpoint, source)) {}
    std::unique_ptr<Connection> connection;
};

RoceClient::RoceClient(std::string_view endpoint, std::string_view source)
    : impl_(std::make_unique<Impl>(endpoint, source)) {}
RoceClient::~RoceClient() = default;

std::string RoceClient::execute(std::string_view request) {
    if (!impl_->connection) {
        throw std::runtime_error("RoCEv2 connection is closed");
    }
    if (request.size() > message_limit) {
        throw std::length_error("RoCEv2 request exceeds the 1 MiB limit");
    }
    try {
        const auto deadline = Clock::now() + request_timeout;
        impl_->connection->send(request, deadline);
        return impl_->connection->receive(deadline);
    } catch (...) {
        impl_->connection.reset();
        throw;
    }
}

void run_roce_server(std::string_view listen, const std::function<std::string(std::string)>& handler) {
    auto channel = make_channel();
    const auto listener = bind_listener(channel.get(), listen);
    while (true) {
        // Keep the child identifier alive until its connection request has been acknowledged.
        Identifier child;
        auto event = await_event(channel.get(), Clock::time_point::max());
        if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
            unexpected_event(*event);
        }
        child.reset(event->id);
        const bool compatible = event->status == 0 && supported_protocol(*event);
        event.reset();
        if (!compatible) {
            rdma_reject(child.get(), nullptr, 0);
            continue;
        }
        try {
            auto session_channel = make_channel();
            check_status(rdma_migrate_id(child.get(), session_channel.get()), "migrate RoCEv2 peer event channel");
            Connection connection(std::move(session_channel), std::move(child));
            connection.establish(false);
            while (true) {
                auto request = connection.receive(Clock::time_point::max());
                const auto response = handler(std::move(request));
                if (response.size() > message_limit) {
                    connection.send("ERR RoCEv2 response exceeds the 1 MiB limit", Clock::now() + request_timeout);
                } else {
                    connection.send(response, Clock::now() + request_timeout);
                }
            }
        } catch (const PeerDisconnected&) {
            continue;
        } catch (const std::exception& error) {
            std::cerr << "RoCEv2 peer: " << error.what() << '\n';
        }
    }
}

} // namespace ef

#else

namespace ef {
namespace {

[[noreturn]] void unsupported() {
    throw std::runtime_error("RoCEv2 is unavailable in this build; enable RDMA=1 with rdma-core");
}

} // namespace

struct RoceClient::Impl {};

RoceClient::RoceClient(std::string_view, std::string_view) { unsupported(); }
RoceClient::~RoceClient() = default;
std::string RoceClient::execute(std::string_view) { unsupported(); }
void run_roce_server(std::string_view, const std::function<std::string(std::string)>&) { unsupported(); }

} // namespace ef

#endif

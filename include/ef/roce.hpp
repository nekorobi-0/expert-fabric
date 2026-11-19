#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace ef {

class RoceClient {
public:
    explicit RoceClient(std::string_view endpoint, std::string_view source_address = {});
    ~RoceClient();

    RoceClient(const RoceClient&) = delete;
    RoceClient& operator=(const RoceClient&) = delete;

    std::string execute(std::string_view request);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void run_roce_server(std::string_view listen, const std::function<std::string(std::string)>& handler);

} // namespace ef

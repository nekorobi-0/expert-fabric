#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ef {

struct NgramPartition {
    std::string_view model;
    std::string_view tokenizer;
    std::string_view sharing_scope;
};

class NgramCache {
public:
    using Token = std::uint32_t;

    // The budget counts stored string and token bytes plus a fixed charge per
    // entry. Allocator and container overhead is outside the logical budget.
    NgramCache(std::size_t min_match, std::size_t max_match, std::size_t max_draft,
               std::size_t max_logical_bytes);
    ~NgramCache();

    NgramCache(const NgramCache&) = delete;
    NgramCache& operator=(const NgramCache&) = delete;

    // Only call after a request is finished, with its finalized token IDs.
    void observe(NgramPartition partition, std::span<const Token> tokens);

    // A returned draft is a candidate for the caller's model to verify.
    [[nodiscard]] std::vector<Token> propose(NgramPartition partition,
                                              std::span<const Token> context) const;
    [[nodiscard]] std::size_t entry_count() const;
    [[nodiscard]] std::size_t logical_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ef

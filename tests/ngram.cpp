#include "ef/ngram.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

using Token = ef::NgramCache::Token;

constexpr ef::NgramPartition local{"model", "tokenizer", "local"};
constexpr ef::NgramPartition remote{"model", "tokenizer", "remote"};

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void lookup() {
    ef::NgramCache cache(2, 3, 2, 8192);
    cache.observe(local, std::array<Token, 7>{7, 1, 2, 11, 12, 13, 14});
    cache.observe(local, std::array<Token, 5>{8, 1, 2, 21, 22});

    check(cache.propose(local, std::array<Token, 4>{9, 7, 1, 2}) == std::vector<Token>{11, 12},
          "the longest exact suffix must win");
    check(cache.propose(local, std::array<Token, 3>{6, 1, 2}) == std::vector<Token>{21, 22},
          "a shorter suffix uses the newest finished request");
    check(cache.propose(local, std::array<Token, 2>{2, 21}) == std::vector<Token>{22},
          "the continuation must end with the learned sequence");
    check(cache.propose(local, std::array<Token, 2>{21, 22}).empty(), "no tokens follow the learned sequence");
    check(cache.propose(local, std::array<Token, 1>{2}).empty(), "too-short context should miss");
    check(cache.propose(remote, std::array<Token, 2>{1, 2}).empty(), "sharing scopes must be isolated");
    check(cache.propose({"other", "tokenizer", "local"}, std::array<Token, 2>{1, 2}).empty(),
          "models must be isolated");
    check(cache.propose({"model", "other", "local"}, std::array<Token, 2>{1, 2}).empty(),
          "tokenizers must be isolated");
    try {
        (void)cache.propose({"model", "tokenizer", ""}, std::array<Token, 2>{1, 2});
        throw std::runtime_error("empty sharing scope should be rejected");
    } catch (const std::invalid_argument&) {
    }
}

void bounds() {
    ef::NgramCache disabled(1, 1, 1, 0);
    disabled.observe(local, std::array<Token, 2>{1, 2});
    check(disabled.entry_count() == 0 && disabled.logical_bytes() == 0, "zero budget must disable storage");

    ef::NgramCache limited(1, 1, 1, 100);
    limited.observe(local, std::array<Token, 2>{1, 2});
    limited.observe(local, std::array<Token, 2>{2, 3});
    check(limited.entry_count() == 1 && limited.logical_bytes() <= 100, "budget must cap retained entries");
    check(limited.propose(local, std::array<Token, 1>{1}).empty(), "oldest entry must be evicted first");
    check(limited.propose(local, std::array<Token, 1>{2}) == std::vector<Token>{3}, "new entry must remain");

    ef::NgramCache large(1, 1, 1, 16 * 1024 * 1024);
    std::vector<Token> tokens;
    for (Token value = 0; value < 20000; ++value) tokens.push_back(value);
    large.observe(local, tokens);
    check(large.entry_count() <= 4096, "one large request must not produce unbounded observations");
    check(large.propose(local, std::array<Token, 1>{19998}) == std::vector<Token>{19999},
          "the newest continuation must survive the observation cap");
}

void concurrent_partitions() {
    ef::NgramCache cache(2, 2, 1, 8192);
    {
        std::jthread observe_a([&] {
            for (int i = 0; i < 300; ++i) cache.observe(local, std::array<Token, 3>{1, 2, 3});
        });
        std::jthread observe_b([&] {
            for (int i = 0; i < 300; ++i) cache.observe(remote, std::array<Token, 3>{1, 2, 4});
        });
        for (int i = 0; i < 300; ++i) {
            auto candidate = cache.propose(local, std::array<Token, 2>{1, 2});
            check(candidate.empty() || candidate == std::vector<Token>{3}, "concurrent lookup crossed a scope");
        }
    }
    check(cache.entry_count() == 2, "repeated observations should replace entries");
    check(cache.propose(remote, std::array<Token, 2>{1, 2}) == std::vector<Token>{4},
          "the simultaneous scope should retain its own continuation");
}

}  // namespace

int main() {
    try {
        lookup();
        bounds();
        concurrent_partitions();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "ef/ngram.hpp"

#include <algorithm>
#include <functional>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ef {
namespace {

using Token = NgramCache::Token;

constexpr std::size_t entry_charge = 64;
constexpr std::size_t max_partition_field = 256;
constexpr std::size_t max_token_window = 64;
constexpr std::size_t max_observations = 4096;

void validate_partition(NgramPartition partition) {
    for (auto field : {partition.model, partition.tokenizer, partition.sharing_scope}) {
        if (field.empty() || field.size() > max_partition_field) {
            throw std::invalid_argument("n-gram cache partition fields must contain 1 to 256 bytes");
        }
    }
}

struct KeyView {
    NgramPartition partition;
    std::span<const Token> suffix;
};

struct Key {
    std::string model;
    std::string tokenizer;
    std::string scope;
    std::vector<Token> suffix;

    explicit Key(KeyView view)
        : model(view.partition.model), tokenizer(view.partition.tokenizer),
          scope(view.partition.sharing_scope), suffix(view.suffix.begin(), view.suffix.end()) {}

    operator KeyView() const noexcept { return {{model, tokenizer, scope}, suffix}; }
};

struct KeyHash {
    using is_transparent = void;

    std::size_t operator()(KeyView key) const noexcept {
        std::size_t hash = 0;
        const auto mix = [&hash](std::size_t value) {
            hash ^= value + std::size_t{0x9e3779b9} + (hash << 6) + (hash >> 2);
        };
        const std::hash<std::string_view> string_hash;
        mix(string_hash(key.partition.model));
        mix(string_hash(key.partition.tokenizer));
        mix(string_hash(key.partition.sharing_scope));
        for (auto token : key.suffix) mix(token);
        return hash;
    }
};

struct KeyEqual {
    using is_transparent = void;

    bool operator()(KeyView lhs, KeyView rhs) const noexcept {
        return lhs.partition.model == rhs.partition.model &&
               lhs.partition.tokenizer == rhs.partition.tokenizer &&
               lhs.partition.sharing_scope == rhs.partition.sharing_scope &&
               std::ranges::equal(lhs.suffix, rhs.suffix);
    }
};

std::size_t logical_cost(KeyView key, std::size_t draft_length) {
    return entry_charge + key.partition.model.size() + key.partition.tokenizer.size() +
           key.partition.sharing_scope.size() + sizeof(Token) * (key.suffix.size() + draft_length);
}

}  // namespace

struct NgramCache::Impl {
    struct Entry {
        std::vector<Token> draft;
        std::list<const Key*>::iterator age;
        std::size_t bytes;
    };

    using Table = std::unordered_map<Key, Entry, KeyHash, KeyEqual>;

    const std::size_t min_match;
    const std::size_t max_match;
    const std::size_t max_draft;
    const std::size_t budget;
    mutable std::mutex mutex;
    Table table;
    std::list<const Key*> newest_first;
    std::size_t bytes = 0;

    Impl(std::size_t min_match, std::size_t max_match, std::size_t max_draft, std::size_t budget)
        : min_match(min_match), max_match(max_match), max_draft(max_draft), budget(budget) {
        if (min_match == 0 || min_match > max_match || max_match > max_token_window ||
            max_draft == 0 || max_draft > max_token_window) {
            throw std::invalid_argument("n-gram match and draft lengths must be 1 to 64 with min <= max");
        }
    }

    void erase(Table::iterator entry) {
        bytes -= entry->second.bytes;
        newest_first.erase(entry->second.age);
        table.erase(entry);
    }

    void remember(KeyView key, std::span<const Token> draft) {
        const auto cost = logical_cost(key, draft.size());
        if (cost > budget) return;

        if (const auto existing = table.find(key); existing != table.end()) erase(existing);
        while (bytes > budget - cost) erase(table.find(*newest_first.back()));

        // List entries point to keys. Rehashing preserves table element addresses.
        auto position = newest_first.insert(newest_first.begin(), nullptr);
        try {
            auto entry = table.emplace(Key(key), Entry{{draft.begin(), draft.end()}, position, cost});
            *position = &entry.first->first;
        } catch (...) {
            newest_first.erase(position);
            throw;
        }
        bytes += cost;
    }
};

NgramCache::NgramCache(std::size_t min_match, std::size_t max_match, std::size_t max_draft,
                       std::size_t max_logical_bytes)
    : impl_(std::make_unique<Impl>(min_match, max_match, max_draft, max_logical_bytes)) {}

NgramCache::~NgramCache() = default;

void NgramCache::observe(NgramPartition partition, std::span<const Token> tokens) {
    validate_partition(partition);
    auto& cache = *impl_;
    if (tokens.size() <= cache.min_match || cache.budget < entry_charge) return;

    // Learn from the most recent positions first when selecting the window,
    // then insert oldest to newest so repeated suffixes keep the newest draft.
    const auto width = cache.max_match - cache.min_match + 1;
    const auto budget_entries = std::min(max_observations, cache.budget / entry_charge);
    const auto positions = std::max(std::size_t{1}, budget_entries / width);
    const auto first = tokens.size() - std::min(positions, tokens.size() - cache.min_match);
    std::lock_guard lock(cache.mutex);
    for (auto split = first; split < tokens.size(); ++split) {
        const auto longest = std::min(cache.max_match, split);
        const auto draft = tokens.subspan(split, std::min(cache.max_draft, tokens.size() - split));
        for (auto length = cache.min_match; length <= longest; ++length) {
            cache.remember({partition, tokens.subspan(split - length, length)}, draft);
        }
    }
}

std::vector<NgramCache::Token> NgramCache::propose(NgramPartition partition,
                                                  std::span<const Token> context) const {
    validate_partition(partition);
    const auto& cache = *impl_;
    std::lock_guard lock(cache.mutex);
    for (auto length = std::min(cache.max_match, context.size()); length >= cache.min_match; --length) {
        if (const auto found = cache.table.find(KeyView{partition, context.last(length)});
            found != cache.table.end()) {
            return found->second.draft;
        }
    }
    return {};
}

std::size_t NgramCache::entry_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->table.size();
}

std::size_t NgramCache::logical_bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->bytes;
}

}  // namespace ef

#include "ef/config.hpp"

#include <yaml.h>

#include <charconv>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace ef {
namespace {

[[noreturn]] void invalid(std::string_view field, std::string_view reason) {
    throw std::runtime_error("configuration " + std::string(field) + ": " + std::string(reason));
}

struct Parser {
    yaml_parser_t value{};

    Parser() {
        if (!yaml_parser_initialize(&value)) throw std::bad_alloc();
    }

    ~Parser() { yaml_parser_delete(&value); }

    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;
};

struct Document {
    yaml_document_t value{};

    explicit Document(Parser& parser) {
        if (!yaml_parser_load(&parser.value, &value)) {
            const auto& mark = parser.value.problem_mark;
            const std::string at = "YAML at line " + std::to_string(mark.line + 1) +
                                   ", column " + std::to_string(mark.column + 1);
            invalid(at, parser.value.problem ? parser.value.problem : "parse failed");
        }
    }

    ~Document() { yaml_document_delete(&value); }

    Document(const Document&) = delete;
    Document& operator=(const Document&) = delete;
};

std::string_view scalar(const yaml_node_t* node, std::string_view field) {
    if (!node || node->type != YAML_SCALAR_NODE) invalid(field, "expected a scalar");
    return {reinterpret_cast<const char*>(node->data.scalar.value), node->data.scalar.length};
}

std::string_view string_value(const yaml_node_t* node, std::string_view field) {
    const auto value = scalar(node, field);
    if (value.empty() || value.find('\0') != std::string_view::npos ||
        (node->data.scalar.style == YAML_PLAIN_SCALAR_STYLE && (value == "null" || value == "~"))) {
        invalid(field, "expected a nonempty string");
    }
    return value;
}

bool bool_value(const yaml_node_t* node, std::string_view field) {
    const auto value = scalar(node, field);
    if (node->data.scalar.style == YAML_PLAIN_SCALAR_STYLE) {
        if (value == "true" || value == "True" || value == "TRUE") return true;
        if (value == "false" || value == "False" || value == "FALSE") return false;
    }
    invalid(field, "expected true or false");
}

template <class Integer>
Integer integer_value(const yaml_node_t* node, std::string_view field) {
    const auto value = scalar(node, field);
    Integer result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (node->data.scalar.style != YAML_PLAIN_SCALAR_STYLE || error != std::errc{} ||
        end != value.data() + value.size()) {
        invalid(field, "expected an unsigned decimal integer in range");
    }
    return result;
}

class Tree {
public:
    explicit Tree(Document& document) : document_(document.value) {}

    const yaml_node_t* root() { return yaml_document_get_root_node(&document_); }

    const yaml_node_t* node(int number) {
        const auto* result = yaml_document_get_node(&document_, number);
        if (!result) invalid("YAML", "invalid node reference");
        return result;
    }

    const yaml_node_t* get(const yaml_node_t* mapping, std::string_view key) {
        if (!mapping) return nullptr;
        if (mapping->type != YAML_MAPPING_NODE) invalid(key, "parent must be a mapping");
        for (auto pair = mapping->data.mapping.pairs.start;
             pair != mapping->data.mapping.pairs.top; ++pair) {
            if (scalar(node(pair->key), "YAML key") == key) return node(pair->value);
        }
        return nullptr;
    }

    void validate() {
        const auto* mapping = root();
        if (!mapping || mapping->type != YAML_MAPPING_NODE) invalid("YAML", "root must be a mapping");
        // Each document node is checked once, even if several anchors refer to it.
        std::vector<unsigned char> marks(static_cast<std::size_t>(document_.nodes.top -
                                                                document_.nodes.start) + 1);
        validate_node(1, marks, 0);
    }

private:
    void validate_node(int number, std::vector<unsigned char>& marks, unsigned depth) {
        const auto* current = node(number);
        auto& mark = marks[static_cast<std::size_t>(number)];
        if (mark == 2) return;
        if (mark == 1) invalid("YAML", "cyclic aliases are unsupported");
        if (depth >= 64) invalid("YAML", "mapping nesting exceeds 64 levels");
        mark = 1;
        if (current->type == YAML_MAPPING_NODE) {
            std::unordered_set<std::string_view> keys;
            for (auto pair = current->data.mapping.pairs.start;
                 pair != current->data.mapping.pairs.top; ++pair) {
                const auto key = string_value(node(pair->key), "YAML key");
                if (key == "<<") invalid("YAML", "merge keys are unsupported");
                if (!keys.insert(key).second) invalid(key, "duplicate key");
                validate_node(pair->value, marks, depth + 1);
            }
        } else if (current->type == YAML_SEQUENCE_NODE) {
            for (auto item = current->data.sequence.items.start;
                 item != current->data.sequence.items.top; ++item) {
                validate_node(*item, marks, depth + 1);
            }
        }
        mark = 2;
    }

    yaml_document_t& document_;
};

void reject_enabled(Tree& tree, const yaml_node_t* mapping, std::string_view key,
                    std::string_view field) {
    if (const auto* node = tree.get(mapping, key); node && bool_value(node, field)) {
        invalid(field, "unsupported by the current PoC");
    }
}

void load_features(Tree& tree, const yaml_node_t* root) {
    if (const auto* active = tree.get(root, "active_profile")) {
        const auto profile_name = string_value(active, "active_profile");
        const auto* selected = tree.get(tree.get(root, "profiles"), profile_name);
        if (!selected || selected->type != YAML_MAPPING_NODE) {
            invalid("active_profile", "must select an existing profile mapping");
        }
        reject_enabled(tree, tree.get(selected, "dspark"), "enabled", "selected profile dspark.enabled");
    }
    reject_enabled(tree, tree.get(root, "dspark"), "enabled", "dspark.enabled");

    const auto* optimization = tree.get(tree.get(root, "main"), "optimization");
    reject_enabled(tree, optimization, "gpu_routing", "main.optimization.gpu_routing");
    reject_enabled(tree, optimization, "hip_graph", "main.optimization.hip_graph");
    if (const auto* streams = tree.get(optimization, "hip_streams");
        streams && integer_value<unsigned>(streams, "main.optimization.hip_streams") != 1) {
        invalid("main.optimization.hip_streams", "multiple HIP streams are unsupported by the current PoC");
    }
}

Config parse_config(Tree& tree) {
    const auto* root = tree.root();
    load_features(tree, root);
    Config result;
    const auto* ngram = tree.get(tree.get(root, "speculation"), "ngram");
    if (const auto* enabled = tree.get(ngram, "enabled");
        enabled && bool_value(enabled, "speculation.ngram.enabled")) {
        auto& cache = result.ngram.emplace();
        if (const auto* node = tree.get(ngram, "min_match_tokens")) {
            cache.min_match = integer_value<std::size_t>(node, "speculation.ngram.min_match_tokens");
        }
        if (const auto* node = tree.get(ngram, "max_match_tokens")) {
            cache.max_match = integer_value<std::size_t>(node, "speculation.ngram.max_match_tokens");
        }
        if (const auto* node = tree.get(ngram, "max_draft_tokens")) {
            cache.max_draft = integer_value<std::size_t>(node, "speculation.ngram.max_draft_tokens");
        }
        if (const auto* node = tree.get(tree.get(ngram, "cache"), "max_ram_mib")) {
            constexpr std::size_t mib = 1024 * 1024;
            const auto limit = integer_value<std::size_t>(node, "speculation.ngram.cache.max_ram_mib");
            if (limit == 0 || limit > std::numeric_limits<std::size_t>::max() / mib) {
                invalid("speculation.ngram.cache.max_ram_mib", "expected a positive representable capacity");
            }
            cache.max_bytes = limit * mib;
        }
    }
    if (const auto* node = tree.get(tree.get(root, "main"), "numa_node")) {
        result.main_numa_node = integer_value<unsigned>(node, "main.numa_node");
    }
    const auto* profiling = tree.get(root, "profiling");
    reject_enabled(tree, profiling, "gpu_timing", "profiling.gpu_timing");
    if (const auto* node = tree.get(profiling, "per_expert_counters")) {
        result.per_expert_counters = bool_value(node, "profiling.per_expert_counters");
    }
    const auto* prometheus = tree.get(profiling, "prometheus");
    if (const auto* node = tree.get(prometheus, "enabled")) {
        result.metrics_enabled = bool_value(node, "profiling.prometheus.enabled");
    }
    if (const auto* node = tree.get(prometheus, "listen")) {
        result.metrics_listen = string_value(node, "profiling.prometheus.listen");
    }

    const auto* engines = tree.get(root, "expert_engines");
    if (!engines || engines->type != YAML_SEQUENCE_NODE ||
        engines->data.sequence.items.top == engines->data.sequence.items.start) {
        invalid("expert_engines", "expected a nonempty engine list");
    }
    result.engines.clear();
    std::unordered_set<std::string> ids;
    for (auto item = engines->data.sequence.items.start; item != engines->data.sequence.items.top; ++item) {
        const auto* engine = tree.node(*item);
        const auto field = "expert_engines[" + std::to_string(result.engines.size()) + "]";
        if (engine->type != YAML_MAPPING_NODE) invalid(field, "expected a mapping");
        auto& target = result.engines.emplace_back();
        if (const auto* node = tree.get(engine, "id")) target.id = string_value(node, field + ".id");
        if (target.id.empty() || target.id.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos ||
            !ids.insert(target.id).second) {
            invalid(field + ".id", "expected a unique identifier using letters, digits, hyphens, underscores or dots");
        }
        if (const auto* node = tree.get(engine, "target_kind")) {
            const auto kind = string_value(node, field + ".target_kind");
            if (kind == "gpu") target.gpu_device = 0;
            else if (kind != "cpu") invalid(field + ".target_kind", "expected cpu or gpu");
        }
        if (const auto* node = tree.get(engine, "gpu_device")) {
            if (!target.gpu_device) invalid(field + ".gpu_device", "requires target_kind gpu");
            target.gpu_device = integer_value<unsigned>(node, field + ".gpu_device");
        }
        if (const auto* node = tree.get(engine, "transport")) {
            target.transport = string_value(node, field + ".transport");
            if (target.transport != "local" && target.transport != "upi" && target.transport != "roce_v2") {
                invalid(field + ".transport", "expected local, upi, or roce_v2");
            }
        }
        if (const auto* node = tree.get(engine, "numa_node")) {
            target.worker_numa_node = integer_value<unsigned>(node, field + ".numa_node");
        }
        if (const auto* node = tree.get(engine, "endpoint")) {
            target.rdma_endpoint = string_value(node, field + ".endpoint");
        }
        if (target.transport == "roce_v2" && target.rdma_endpoint.empty()) {
            invalid(field + ".endpoint", "required for roce_v2");
        }
        if (const auto* node = tree.get(engine, "source_address")) {
            target.rdma_source_address = string_value(node, field + ".source_address");
            if (target.transport != "roce_v2" || target.rdma_source_address.empty()) {
                invalid(field + ".source_address", "requires a RoCE v2 engine and a nonempty local IP address");
            }
        }
        if (const auto* node = tree.get(engine, "cpu_vectorize")) {
            target.cpu_vectorize = bool_value(node, field + ".cpu_vectorize");
            if (target.gpu_device && target.cpu_vectorize) invalid(field + ".cpu_vectorize", "requires target_kind cpu");
        }
        if (const auto* node = tree.get(engine, "capacity_bytes")) {
            target.capacity_bytes = integer_value<std::size_t>(node, field + ".capacity_bytes");
            if (target.capacity_bytes == 0) invalid(field + ".capacity_bytes", "must be positive");
        }
        if (const auto* node = tree.get(engine, "bandwidth_bytes_per_second")) {
            target.bandwidth_bytes_per_second =
                integer_value<std::size_t>(node, field + ".bandwidth_bytes_per_second");
            if (*target.bandwidth_bytes_per_second == 0) invalid(field + ".bandwidth_bytes_per_second", "must be positive");
        }
    }
    return result;
}

}  // namespace

Config load_config(const std::filesystem::path& path) {
    const auto close_file = [](std::FILE* file) { std::fclose(file); };
    const std::unique_ptr<std::FILE, decltype(close_file)> file(std::fopen(path.c_str(), "rb"), close_file);
    if (!file) invalid(path.string(), "cannot open file");

    Parser parser;
    yaml_parser_set_input_file(&parser.value, file.get());
    Document document(parser);
    Tree tree(document);
    tree.validate();
    Document trailing(parser);
    if (yaml_document_get_root_node(&trailing.value)) invalid("YAML", "expected one document");
    return parse_config(tree);
}

}  // namespace ef

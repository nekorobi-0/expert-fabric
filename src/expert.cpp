#include "ef/expert.hpp"

#ifdef EF_HAVE_HIP
#include "ef/hip_expert.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define EF_AVX2_DISPATCH 1
#endif

namespace ef {
namespace {

using Clock = std::chrono::steady_clock;

enum class State { loading, ready, active, draining, retired };

struct WeightShape {
    std::size_t input;
    std::size_t hidden;
    std::size_t matrix_values;
    std::size_t payload_bytes;
};

struct Generation {
    explicit Generation(WeightShape dimensions) : shape(dimensions) {}

    WeightShape shape;
    std::vector<float> weights;
#ifdef EF_HAVE_HIP
    std::unique_ptr<HipExpert> gpu;
#endif
    State state{State::loading};
    std::size_t in_flight{};
    bool retire_requested{};
};

struct Expert {
    std::map<std::uint64_t, Generation> generations;
    std::optional<std::uint64_t> current;
};

bool read_word(std::istream& in, std::string& word) {
    constexpr int max_length = 64;
    if (!(in >> std::setw(max_length) >> word)) {
        return false;
    }
    if (word.size() < max_length) {
        return true;
    }
    const auto next = in.peek();
    return next == std::char_traits<char>::eof() || next == ' ' || (next >= '\t' && next <= '\r');
}

std::size_t read_dimension(std::istream& in) {
    std::string word;
    std::size_t value{};
    if (!read_word(in, word)) {
        throw std::runtime_error("LOAD_FAILED: missing or oversized matrix dimension");
    }
    const auto [ptr, ec] = std::from_chars(word.data(), word.data() + word.size(), value);
    if (ec != std::errc{} || ptr != word.data() + word.size() || value == 0) {
        throw std::runtime_error("LOAD_FAILED: invalid matrix dimension");
    }
    return value;
}

WeightShape read_shape(std::istream& in) {
    std::string magic;
    if (!read_word(in, magic) || magic != "EFMLP1") {
        throw std::runtime_error("LOAD_FAILED: expected EFMLP1 weight header");
    }
    const auto input = read_dimension(in);
    const auto hidden = read_dimension(in);
    const auto max_values = std::vector<float>{}.max_size() / 3;
    const auto max_payload_values = std::numeric_limits<std::size_t>::max() / (3 * sizeof(float));
    if (input > std::min(max_values, max_payload_values) / hidden) {
        throw std::runtime_error("LOAD_FAILED: matrix dimensions are too large");
    }
    return {input, hidden, input * hidden, input * hidden * 3 * sizeof(float)};
}

std::vector<float> read_values(std::istream& in, std::size_t count) {
    std::vector<float> weights(count);
    std::string word;
    for (float& weight : weights) {
        if (!read_word(in, word)) {
            throw std::runtime_error("LOAD_FAILED: missing or oversized weight value");
        }
        const auto begin = word.data() + (word.front() == '+');
        const auto end = word.data() + word.size();
        const auto [ptr, ec] = std::from_chars(begin, end, weight);
        if (ec != std::errc{} || ptr != end || !std::isfinite(weight) || word.starts_with("+-")) {
            throw std::runtime_error("LOAD_FAILED: invalid or non-finite weight value");
        }
    }
    in >> std::ws;
    if (!in.eof() || in.bad()) {
        throw std::runtime_error("LOAD_FAILED: trailing data or input error in weights");
    }
    return weights;
}

double silu(double value) {
    if (value >= 0) {
        return value / (1 + std::exp(-value));
    }
    const double exp = std::exp(value);
    return value * exp / (1 + exp);
}

std::pair<double, double> gate_up_scalar(const float* gate, const float* up,
                                         const float* input, std::size_t count) {
    double gate_value = 0;
    double up_value = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double value = input[i];
        gate_value += gate[i] * value;
        up_value += up[i] * value;
    }
    return {gate_value, up_value};
}

double down_scalar(const float* weights, const double* hidden, std::size_t count) {
    double sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += weights[i] * hidden[i];
    }
    return sum;
}

#if defined(EF_AVX2_DISPATCH)
__attribute__((target("avx2,no-fma")))
double sum_lanes(__m256d values) {
    const auto pair = _mm_add_pd(_mm256_castpd256_pd128(values), _mm256_extractf128_pd(values, 1));
    return _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
}

__attribute__((target("avx2,no-fma")))
std::pair<double, double> gate_up_avx2(const float* gate, const float* up,
                                       const float* input, std::size_t count) {
    auto gate_sum = _mm256_setzero_pd();
    auto up_sum = _mm256_setzero_pd();
    const auto blocks_end = count - count % 4;
    for (std::size_t i = 0; i < blocks_end; i += 4) {
        const auto x = _mm256_cvtps_pd(_mm_loadu_ps(input + i));
        const auto g = _mm256_cvtps_pd(_mm_loadu_ps(gate + i));
        const auto u = _mm256_cvtps_pd(_mm_loadu_ps(up + i));
        gate_sum = _mm256_add_pd(gate_sum, _mm256_mul_pd(g, x));
        up_sum = _mm256_add_pd(up_sum, _mm256_mul_pd(u, x));
    }
    auto gate_value = sum_lanes(gate_sum);
    auto up_value = sum_lanes(up_sum);
    for (std::size_t i = blocks_end; i < count; ++i) {
        const double value = input[i];
        gate_value += gate[i] * value;
        up_value += up[i] * value;
    }
    return {gate_value, up_value};
}

__attribute__((target("avx2,no-fma")))
double down_avx2(const float* weights, const double* hidden, std::size_t count) {
    auto sum = _mm256_setzero_pd();
    const auto blocks_end = count - count % 4;
    for (std::size_t i = 0; i < blocks_end; i += 4) {
        const auto w = _mm256_cvtps_pd(_mm_loadu_ps(weights + i));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(w, _mm256_loadu_pd(hidden + i)));
    }
    auto result = sum_lanes(sum);
    for (std::size_t i = blocks_end; i < count; ++i) {
        result += weights[i] * hidden[i];
    }
    return result;
}
#endif

bool can_vectorize(bool requested) noexcept {
#if defined(EF_AVX2_DISPATCH)
    return requested && __builtin_cpu_supports("avx2");
#else
    (void)requested;
    return false;
#endif
}

template <auto GateUp, auto Down>
std::vector<float> evaluate(const Generation& generation, std::span<const float> input) {
    const auto width = generation.shape.input;
    const auto hidden_width = generation.shape.hidden;
    const auto matrix_values = generation.shape.matrix_values;
    const auto* gate = generation.weights.data();
    const auto* up = gate + matrix_values;
    const auto* down = up + matrix_values;
    std::vector<float> output(input.size());
    std::vector<double> hidden(hidden_width);

    for (std::size_t offset = 0; offset < input.size(); offset += width) {
        for (std::size_t row = 0; row < hidden_width; ++row) {
            const auto [gate_value, up_value] = GateUp(gate + row * width, up + row * width,
                                                      input.data() + offset, width);
            hidden[row] = silu(gate_value) * up_value;
        }
        for (std::size_t row = 0; row < width; ++row) {
            const double value = Down(down + row * hidden_width, hidden.data(), hidden_width);
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
                throw std::runtime_error("COMPUTE_FAILED: expert output does not fit float32");
            }
            output[offset + row] = static_cast<float>(value);
        }
    }
    return output;
}

std::vector<float> run_expert(const Generation& generation, std::span<const float> input, bool vectorized) {
#if defined(EF_AVX2_DISPATCH)
    if (vectorized) {
        return evaluate<gate_up_avx2, down_avx2>(generation, input);
    }
#else
    (void)vectorized;
#endif
    return evaluate<gate_up_scalar, down_scalar>(generation, input);
}

}  // namespace

struct ExpertEngine::Impl {
    Impl(std::size_t capacity_bytes, bool vectorize, std::optional<unsigned> gpu_device)
        : capacity(capacity_bytes), vectorized(can_vectorize(vectorize)), device(gpu_device) {
#ifndef EF_HAVE_HIP
        if (device) throw std::runtime_error("UNSUPPORTED_TARGET: GPU backend requires HIP=1");
#endif
    }

    struct Invocation {
        Impl& engine;
        Generation& generation;

        ~Invocation() {
            const std::lock_guard lock(engine.mutex);
            --generation.in_flight;
            engine.release_if_drained(generation);
        }
    };

    Expert& find_expert(const std::string& id) {
        const auto it = experts.find(id);
        if (it == experts.end()) {
            throw std::runtime_error("NOT_READY: unknown expert");
        }
        return it->second;
    }

    static Generation& find_generation(Expert& expert, std::uint64_t generation) {
        const auto it = expert.generations.find(generation);
        if (it == expert.generations.end()) {
            throw std::runtime_error("NOT_READY: unknown expert generation");
        }
        return it->second;
    }

    void release_if_drained(Generation& entry) {
        if (entry.state == State::draining && entry.retire_requested && entry.in_flight == 0) {
            std::vector<float>{}.swap(entry.weights);
#ifdef EF_HAVE_HIP
            entry.gpu.reset();
#endif
            used -= entry.shape.payload_bytes;
            entry.state = State::retired;
        }
    }

    const std::size_t capacity;
    const bool vectorized;
    const std::optional<unsigned> device;
    std::size_t used{};
    std::mutex mutex;
    std::map<std::string, Expert> experts;
};

ExpertEngine::ExpertEngine(std::size_t capacity_bytes, bool vectorize, std::optional<unsigned> gpu_device)
    : impl_(std::make_unique<Impl>(capacity_bytes, vectorize, gpu_device)) {}

ExpertEngine::~ExpertEngine() = default;

std::size_t ExpertEngine::capacity_bytes() const noexcept {
    return impl_->capacity;
}

bool ExpertEngine::vectorized() const noexcept {
    return impl_->vectorized;
}

bool ExpertEngine::gpu_enabled() const noexcept {
    return impl_->device.has_value();
}

std::size_t ExpertEngine::available_bytes() const {
    const std::lock_guard lock(impl_->mutex);
    return impl_->capacity - impl_->used;
}

std::string ExpertEngine::status(const std::string& expert, std::uint64_t generation) const {
    const std::lock_guard lock(impl_->mutex);
    const auto model = impl_->experts.find(expert);
    if (model == impl_->experts.end()) {
        return "ABSENT";
    }
    const auto entry = model->second.generations.find(generation);
    if (entry == model->second.generations.end()) {
        return "ABSENT";
    }
    switch (entry->second.state) {
        case State::loading: return "LOADING";
        case State::ready: return "READY";
        case State::active: return "ACTIVE";
        case State::draining: return "DRAINING";
        case State::retired: return "RETIRED";
    }
    throw std::logic_error("invalid expert state");
}

void ExpertEngine::load(const std::string& expert, std::uint64_t generation,
                        const std::filesystem::path& weights) {
    if (expert.empty()) {
        throw std::runtime_error("LOAD_FAILED: expert ID must be non-empty");
    }
    std::ifstream in(weights);
    in.imbue(std::locale::classic());
    if (!in) {
        throw std::runtime_error("LOAD_FAILED: cannot open weight file " + weights.string());
    }
    const auto shape = read_shape(in);

    {
        const std::lock_guard lock(impl_->mutex);
        auto& entry = impl_->experts[expert];
        if (entry.generations.contains(generation)) {
            throw std::runtime_error("STALE_GENERATION: generation already exists");
        }
        if (shape.payload_bytes > impl_->capacity - impl_->used) {
            throw std::runtime_error("NO_CAPACITY: not enough room for weights");
        }
        entry.generations.try_emplace(generation, shape);
        impl_->used += shape.payload_bytes;
    }

    try {
        auto values = read_values(in, shape.matrix_values * 3);
#ifdef EF_HAVE_HIP
        auto gpu = impl_->device ? std::make_unique<HipExpert>(*impl_->device, shape.input, shape.hidden, values) : nullptr;
#endif
        const std::lock_guard lock(impl_->mutex);
        auto& entry = Impl::find_generation(impl_->find_expert(expert), generation);
#ifdef EF_HAVE_HIP
        if (gpu) entry.gpu = std::move(gpu);
        else
#endif
        entry.weights = std::move(values);
        entry.state = State::ready;
    } catch (...) {
        const std::lock_guard lock(impl_->mutex);
        auto& generations = impl_->find_expert(expert).generations;
        generations.erase(generation);
        impl_->used -= shape.payload_bytes;
        throw;
    }
}

void ExpertEngine::publish(const std::string& expert, std::uint64_t generation,
                           std::optional<std::uint64_t> expected) {
    const std::lock_guard lock(impl_->mutex);
    auto& model = impl_->find_expert(expert);
    auto& entry = Impl::find_generation(model, generation);
    if (model.current != expected || entry.state == State::draining || entry.state == State::retired) {
        throw std::runtime_error("STALE_GENERATION: publish generation or expectation is stale");
    }
    if (entry.state != State::ready) {
        throw std::runtime_error("NOT_READY: publish requires a ready generation");
    }
    if (model.current) {
        Impl::find_generation(model, *model.current).state = State::draining;
    }
    entry.state = State::active;
    model.current = generation;
}

std::vector<float> ExpertEngine::invoke(const std::string& expert, std::uint64_t generation,
                                         std::span<const float> input, InvocationStats* stats) {
    const auto queued_at = stats ? Clock::now() : Clock::time_point{};
    auto& entry = [&]() -> Generation& {
        const std::lock_guard lock(impl_->mutex);
        auto& state = Impl::find_generation(impl_->find_expert(expert), generation);
        if (state.state == State::draining || state.state == State::retired) {
            throw std::runtime_error("STALE_GENERATION: expert generation is no longer active");
        }
        if (state.state != State::active) {
            throw std::runtime_error("NOT_READY: expert generation is not active");
        }
        if (input.empty() || input.size() % state.shape.input != 0) {
            throw std::runtime_error("INVALID_BUFFER: input must contain whole tokens");
        }
        ++state.in_flight;
        return state;
    }();

    const Impl::Invocation invocation{*impl_, entry};
    const auto start = stats ? Clock::now() : Clock::time_point{};
    if (!std::ranges::all_of(input, [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("INVALID_BUFFER: input contains a non-finite value");
    }
#ifdef EF_HAVE_HIP
    auto result = entry.gpu ? entry.gpu->invoke(input) : run_expert(entry, input, impl_->vectorized);
#else
    auto result = run_expert(entry, input, impl_->vectorized);
#endif
    if (stats != nullptr) {
        *stats = {input.size() / entry.shape.input,
                  std::chrono::duration<double>(start - queued_at).count(),
                  std::chrono::duration<double>(Clock::now() - start).count()};
    }
    return result;
}

void ExpertEngine::retire(const std::string& expert, std::uint64_t generation) {
    const std::lock_guard lock(impl_->mutex);
    auto& model = impl_->find_expert(expert);
    auto& entry = Impl::find_generation(model, generation);
    if (entry.state == State::loading) {
        throw std::runtime_error("NOT_READY: cannot retire a generation that is loading");
    }
    if (entry.state == State::retired) {
        return;
    }
    if (model.current == generation) {
        model.current.reset();
    }
    entry.state = State::draining;
    entry.retire_requested = true;
    impl_->release_if_drained(entry);
}

}  // namespace ef

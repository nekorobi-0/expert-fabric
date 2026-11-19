#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ef {

struct InvocationStats {
    std::size_t tokens{};
    double queue_seconds{};
    double compute_seconds{};
};

class ExpertEngine {
public:
    // AVX2, when available, changes summation order. Use scalar for bitwise reference runs.
    explicit ExpertEngine(std::size_t capacity_bytes, bool vectorize = false,
                          std::optional<unsigned> gpu_device = std::nullopt);
    ~ExpertEngine();

    ExpertEngine(const ExpertEngine&) = delete;
    ExpertEngine& operator=(const ExpertEngine&) = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t available_bytes() const;
    [[nodiscard]] bool vectorized() const noexcept;
    [[nodiscard]] bool gpu_enabled() const noexcept;
    [[nodiscard]] std::string status(const std::string& expert, std::uint64_t generation) const;

    // EFMLP1 files carry row-major gate, up, and down matrices.
    void load(const std::string& expert, std::uint64_t generation,
              const std::filesystem::path& weights);
    void publish(const std::string& expert, std::uint64_t generation,
                 std::optional<std::uint64_t> expected);

    // Input and output consist of consecutive tokens of the model's input width.
    [[nodiscard]] std::vector<float> invoke(const std::string& expert, std::uint64_t generation,
                                            std::span<const float> input,
                                            InvocationStats* stats = nullptr);

    // Stops admission; the last in-flight invocation releases the weights.
    void retire(const std::string& expert, std::uint64_t generation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ef

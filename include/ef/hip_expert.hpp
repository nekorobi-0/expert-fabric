#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ef {

class HipExpert {
public:
    // Matrices are row-major gate and up (hidden × input), then down (input × hidden).
    HipExpert(unsigned device, std::size_t input_dim, std::size_t hidden_dim,
              std::span<const float> packed_weights);
    ~HipExpert();

    HipExpert(const HipExpert&) = delete;
    HipExpert& operator=(const HipExpert&) = delete;

    // A batch contains one or more contiguous input vectors. Calls on the same
    // instance are serialized, and the returned output is ready on the host.
    [[nodiscard]] std::vector<float> invoke(std::span<const float> input) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ef

#include "ef/hip_expert.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ef {
namespace {

constexpr unsigned threads_per_block = 128;

void check(hipError_t result, std::string_view phase, std::string_view operation) {
    if (result == hipSuccess) return;
    throw std::runtime_error(std::string(phase) + ": " + std::string(operation) + ": " +
                             hipGetErrorString(result));
}

class DeviceScope {
public:
    DeviceScope(int device, std::string_view phase) {
        check(hipGetDevice(&previous_), phase, "get HIP device");
        if (device != previous_) {
            check(hipSetDevice(device), phase, "select HIP device");
            changed_ = true;
        }
    }

    ~DeviceScope() {
        if (changed_) (void)hipSetDevice(previous_);
    }

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_{};
    bool changed_{};
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check(hipMalloc(&data_, bytes), "COMPUTE_FAILED", "allocate HIP scratch");
    }

    ~DeviceBuffer() { (void)hipFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const { return data_; }

private:
    void* data_{};
};

class StreamDrain {
public:
    explicit StreamDrain(hipStream_t stream) : stream_(stream) {}

    ~StreamDrain() {
        if (stream_) (void)hipStreamSynchronize(stream_);
    }

    void wait() {
        check(hipStreamSynchronize(stream_), "COMPUTE_FAILED", "wait for HIP expert output");
        stream_ = nullptr;
    }

private:
    hipStream_t stream_;
};

unsigned block_count(std::size_t values) {
    return static_cast<unsigned>(std::min<std::size_t>(1 + (values - 1) / threads_per_block, 65535));
}

__global__ void gate_up_kernel(const float* weights, const float* input, double* activations,
                               std::size_t width, std::size_t hidden, std::size_t total) {
    const auto* up = weights + width * hidden;
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const auto row_start = (i % hidden) * width;
        const auto token_start = (i / hidden) * width;
        double gate_sum = 0.0;
        double up_sum = 0.0;
        for (std::size_t j = 0; j < width; ++j) {
            const double value = input[token_start + j];
            gate_sum += weights[row_start + j] * value;
            up_sum += up[row_start + j] * value;
        }
        const double z = exp(-fabs(gate_sum));
        const double silu = gate_sum >= 0 ? gate_sum / (1 + z) : gate_sum * z / (1 + z);
        activations[i] = silu * up_sum;
    }
}

__global__ void down_kernel(const float* weights, const double* activations, float* output,
                            std::size_t width, std::size_t hidden, std::size_t total) {
    const auto* down = weights + 2 * width * hidden;
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const auto row_start = (i % width) * hidden;
        const auto token_start = (i / width) * hidden;
        double sum = 0.0;
        for (std::size_t j = 0; j < hidden; ++j) {
            sum += down[row_start + j] * activations[token_start + j];
        }
        output[i] = fabs(sum) > std::numeric_limits<float>::max()
                        ? std::numeric_limits<float>::infinity() : static_cast<float>(sum);
    }
}

}  // namespace

struct HipExpert::Impl {
    int device{};
    std::size_t width{};
    std::size_t hidden{};
    hipStream_t stream{};
    float* weights{};
    mutable std::mutex invoke_mutex;

    ~Impl() {
        if (!stream && !weights) return;
        int previous{};
        const bool known = hipGetDevice(&previous) == hipSuccess;
        if ((!known || previous != device) && hipSetDevice(device) != hipSuccess) return;
        if (stream) (void)hipStreamDestroy(stream);
        if (weights) (void)hipFree(weights);
        if (known && previous != device) (void)hipSetDevice(previous);
    }
};

HipExpert::HipExpert(unsigned device, std::size_t input_dim, std::size_t hidden_dim,
                     std::span<const float> packed_weights)
    : impl_(std::make_unique<Impl>()) {
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / (3 * sizeof(float));
    if (input_dim == 0 || hidden_dim == 0 || input_dim > limit / hidden_dim ||
        packed_weights.size() != 3 * input_dim * hidden_dim) {
        throw std::runtime_error("LOAD_FAILED: invalid HIP expert weight shape");
    }
    if (!std::ranges::all_of(packed_weights, [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LOAD_FAILED: non-finite HIP expert weight");
    }

    int count{};
    check(hipGetDeviceCount(&count), "LOAD_FAILED", "enumerate HIP devices");
    if (count <= 0 || device >= static_cast<unsigned>(count)) {
        throw std::runtime_error("LOAD_FAILED: HIP device index is out of range");
    }
    impl_->device = static_cast<int>(device);
    impl_->width = input_dim;
    impl_->hidden = hidden_dim;
    DeviceScope scope(impl_->device, "LOAD_FAILED");
    check(hipStreamCreateWithFlags(&impl_->stream, hipStreamNonBlocking), "LOAD_FAILED", "create HIP stream");
    check(hipMalloc(&impl_->weights, packed_weights.size_bytes()), "LOAD_FAILED", "allocate HIP expert weights");
    check(hipMemcpyAsync(impl_->weights, packed_weights.data(), packed_weights.size_bytes(),
                         hipMemcpyHostToDevice, impl_->stream), "LOAD_FAILED", "upload HIP expert weights");
    check(hipStreamSynchronize(impl_->stream), "LOAD_FAILED", "wait for HIP expert weights");
}

HipExpert::~HipExpert() = default;

std::vector<float> HipExpert::invoke(std::span<const float> input) const {
    if (input.empty() || input.size() % impl_->width != 0 ||
        !std::ranges::all_of(input, [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("INVALID_BUFFER: expected finite whole HIP expert input tokens");
    }
    const auto tokens = input.size() / impl_->width;
    constexpr auto activation_limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (tokens > activation_limit / impl_->hidden || input.size() > std::vector<float>{}.max_size()) {
        throw std::runtime_error("INVALID_BUFFER: HIP expert input is too large");
    }

    const auto hidden_values = tokens * impl_->hidden;
    std::lock_guard lock(impl_->invoke_mutex);
    DeviceScope scope(impl_->device, "COMPUTE_FAILED");
    std::vector<float> output(input.size());
    DeviceBuffer device_input(input.size_bytes());
    DeviceBuffer device_output(input.size_bytes());
    DeviceBuffer activations(hidden_values * sizeof(double));
    StreamDrain drain(impl_->stream);
    check(hipMemcpyAsync(device_input.get(), input.data(), input.size_bytes(), hipMemcpyHostToDevice, impl_->stream),
          "COMPUTE_FAILED", "copy HIP expert input");

    gate_up_kernel<<<block_count(hidden_values), threads_per_block, 0, impl_->stream>>>(
        impl_->weights, static_cast<const float*>(device_input.get()), static_cast<double*>(activations.get()),
        impl_->width, impl_->hidden, hidden_values);
    check(hipGetLastError(), "COMPUTE_FAILED", "launch HIP gate/up kernel");
    down_kernel<<<block_count(input.size()), threads_per_block, 0, impl_->stream>>>(
        impl_->weights, static_cast<const double*>(activations.get()), static_cast<float*>(device_output.get()),
        impl_->width, impl_->hidden, input.size());
    check(hipGetLastError(), "COMPUTE_FAILED", "launch HIP down kernel");
    check(hipMemcpyAsync(output.data(), device_output.get(), input.size_bytes(), hipMemcpyDeviceToHost, impl_->stream),
          "COMPUTE_FAILED", "copy HIP expert output");
    drain.wait();
    if (!std::ranges::all_of(output, [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("COMPUTE_FAILED: HIP expert output does not fit float32");
    }
    return output;
}

}  // namespace ef

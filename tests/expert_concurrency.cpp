#include "ef/expert.hpp"

#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

void require(bool condition, std::string_view description) {
    if (!condition) {
        throw std::runtime_error(std::string(description));
    }
}

struct TestDirectory {
    std::filesystem::path path;

    TestDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "ef-expert-test-XXXXXX").string();
        require(::mkdtemp(pattern.data()) != nullptr, "could not create test directory");
        path = pattern;
    }

    ~TestDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

void pending_load_reserves_capacity(const std::filesystem::path& fifo,
                                    const std::filesystem::path& weights) {
    require(::mkfifo(fifo.c_str(), 0600) == 0, "could not create weight source");
    ef::ExpertEngine engine(3 * sizeof(float));
    auto pending = std::async(std::launch::async, [&] { engine.load("expert", 1, fifo); });

    std::ofstream writer(fifo);
    writer << "EFMLP1 1 1\n" << std::flush;
    require(writer.good(), "could not send weight header");

    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (engine.status("expert", 1) != "LOADING") {
        require(std::chrono::steady_clock::now() < deadline, "LOAD did not reserve capacity");
        std::this_thread::sleep_for(1ms);
    }
    require(engine.available_bytes() == 0, "pending weight payload was not reserved");
    try {
        engine.load("expert", 2, weights);
        throw std::runtime_error("accepted another generation over capacity");
    } catch (const std::runtime_error& error) {
        require(std::string_view(error.what()).starts_with("NO_CAPACITY:"), error.what());
    }
    writer << "1 2 3\n";
    writer.close();
    pending.get();
    require(engine.status("expert", 1) == "READY", "loaded generation is not ready");
}

void only_one_publish_can_replace_a_generation(const std::filesystem::path& weights) {
    ef::ExpertEngine engine(9 * sizeof(float));
    for (const auto generation : {1u, 2u, 3u}) {
        engine.load("expert", generation, weights);
    }
    engine.publish("expert", 1, std::nullopt);

    std::barrier start(2);
    const auto publish = [&](std::uint64_t generation) {
        start.arrive_and_wait();
        try {
            engine.publish("expert", generation, 1);
            return true;
        } catch (const std::runtime_error& error) {
            require(std::string_view(error.what()).starts_with("STALE_GENERATION:"), error.what());
            return false;
        }
    };
    auto second = std::async(std::launch::async, publish, 2);
    const bool third_won = publish(3);
    const bool second_won = second.get();
    require(second_won != third_won, "exactly one PUBLISH must replace generation 1");
    require(engine.status("expert", 1) == "DRAINING", "replaced generation is not draining");
    require(engine.status("expert", second_won ? 2 : 3) == "ACTIVE", "winner is not active");
    require(engine.status("expert", second_won ? 3 : 2) == "READY", "loser changed state");
    engine.retire("expert", 1);
    require(engine.status("expert", 1) == "RETIRED", "old generation was not retired");
    require(engine.available_bytes() == 3 * sizeof(float), "retire did not reclaim weights");
}

void avx2_matches_reference_with_remainders(const std::filesystem::path& path) {
    constexpr std::size_t width = 11;
    constexpr std::size_t hidden_width = 7;
    constexpr auto matrix_size = width * hidden_width;
    std::vector<float> weights(matrix_size * 3);
    std::vector<float> input(3 * width);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const auto signed_value = static_cast<int>((i * 13 + 7) % 43) - 21;
        weights[i] = std::ldexp(static_cast<float>(signed_value), -static_cast<int>(i % 9) - 2);
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>((i * 7 + 5) % 19) - 9) / 4;
    }
    {
        std::ofstream file(path);
        file << "EFMLP1 " << width << ' ' << hidden_width << '\n'
             << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (const auto value : weights) {
            file << value << ' ';
        }
    }
    ef::ExpertEngine scalar(weights.size() * sizeof(float));
    ef::ExpertEngine selected(weights.size() * sizeof(float), true);
    require(!scalar.vectorized(), "AVX2 was used without an opt-in");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    require(selected.vectorized() == static_cast<bool>(__builtin_cpu_supports("avx2")),
            "AVX2 selection does not match runtime capabilities");
#else
    require(!selected.vectorized(), "AVX2 was selected for a portable build");
#endif
    const auto run = [&](ef::ExpertEngine& engine) {
        engine.load("expert", 1, path);
        engine.publish("expert", 1, std::nullopt);
        return engine.invoke("expert", 1, input);
    };
    const auto scalar_output = run(scalar);
    const auto selected_output = run(selected);
    require(scalar_output.size() == input.size() && selected_output.size() == input.size(),
            "expert output did not preserve batch shape");

    for (std::size_t token = 0; token < 3; ++token) {
        long double hidden[hidden_width]{};
        for (std::size_t h = 0; h < hidden_width; ++h) {
            long double gate = 0;
            long double up = 0;
            for (std::size_t col = 0; col < width; ++col) {
                const long double x = input[token * width + col];
                gate += x * weights[h * width + col];
                up += x * weights[matrix_size + h * width + col];
            }
            hidden[h] = gate * up / (1 + std::exp(-gate));
        }
        for (std::size_t row = 0; row < width; ++row) {
            long double reference = 0;
            for (std::size_t col = 0; col < hidden_width; ++col) {
                reference += hidden[col] * weights[matrix_size * 2 + row * hidden_width + col];
            }
            const auto tolerance = 4 * std::numeric_limits<float>::epsilon() * std::abs(reference) + 1e-10L;
            const auto index = token * width + row;
            require(std::abs(scalar_output[index] - reference) <= tolerance, "scalar differs from reference");
            require(std::abs(selected_output[index] - reference) <= tolerance, "AVX2 differs from reference");
        }
    }

    input[width] = std::numeric_limits<float>::infinity();
    for (auto* engine : {&scalar, &selected}) {
        try {
            (void)engine->invoke("expert", 1, input);
            throw std::runtime_error("accepted a non-finite activation");
        } catch (const std::runtime_error& error) {
            require(std::string_view(error.what()).starts_with("INVALID_BUFFER:"), error.what());
        }
    }
}

}  // namespace

int main() {
    try {
        const TestDirectory directory;
        const auto weights = directory.path / "weights.efmlp";
        std::ofstream(weights) << "EFMLP1 1 1\n1 2 3\n";
        pending_load_reserves_capacity(directory.path / "weight-pipe", weights);
        only_one_publish_can_replace_a_generation(weights);
        avx2_matches_reference_with_remainders(directory.path / "numeric.efmlp");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

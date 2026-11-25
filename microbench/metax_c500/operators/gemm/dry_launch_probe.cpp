/* Host-side mcBLAS GEMM selector probe used with dry_launch_interposer.cpp. */

#include <mc_runtime.h>
#include "mcblas.h"

#include <dlfcn.h>

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>


static_assert(
    static_cast<int>(MCBLAS_GEMM_DEFAULT) == -1,
    "the dry-launch probe requires the documented default heuristic value -1");


namespace {

using MarkerBegin = int (*)(std::size_t);
using MarkerEnd = int (*)();
using MarkerError = const char* (*)();

[[noreturn]] void Fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(EXIT_FAILURE);
}

template <typename Function>
Function require_interposer_symbol(const char* name) {
    dlerror();
    void* symbol = dlsym(RTLD_DEFAULT, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        std::fprintf(stderr, "required interposer symbol %s is absent: %s\n",
                     name, error == nullptr ? "null symbol" : error);
        std::exit(EXIT_FAILURE);
    }
    return reinterpret_cast<Function>(symbol);
}

int positive_environment_int(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        std::fprintf(stderr, "%s is required\n", name);
        std::exit(EXIT_FAILURE);
    }
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0
            || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive int: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int positive_int(const std::string& text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || value <= 0
            || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive int: %s\n", name,
                     text.c_str());
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

std::size_t checked_elements(int rows, int columns, const char* label) {
    const auto left = static_cast<std::size_t>(rows);
    const auto right = static_cast<std::size_t>(columns);
    if (left > std::numeric_limits<std::size_t>::max() / right) {
        std::fprintf(stderr, "%s element count overflows size_t\n", label);
        std::exit(EXIT_FAILURE);
    }
    return left * right;
}

void check_runtime(mcError_t status, const char* operation) {
    if (status != mcSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     mcGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void check_mcblas(mcblasStatus_t status, const char* operation) {
    if (status != MCBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     mcblasGetStatusString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct Buffers {
    void* input = nullptr;
    void* weight = nullptr;
    void* result = nullptr;
    std::size_t input_bytes = 0;
    std::size_t weight_bytes = 0;
    std::size_t result_bytes = 0;

    ~Buffers() {
        if (result != nullptr) check_runtime(mcFree(result), "mcFree(result)");
        if (weight != nullptr) check_runtime(mcFree(weight), "mcFree(weight)");
        if (input != nullptr) check_runtime(mcFree(input), "mcFree(input)");
    }

    void ensure(void** pointer, std::size_t* capacity, std::size_t required,
                const char* label) {
        if (*capacity >= required) return;
        if (*pointer != nullptr) check_runtime(mcFree(*pointer), label);
        check_runtime(mcMalloc(pointer, required), label);
        *capacity = required;
    }

    void prepare(int m, int n, int k) {
        constexpr std::size_t element_bytes = 2;
        ensure(&input, &input_bytes,
               checked_elements(m, k, "input") * element_bytes,
               "mcMalloc(input)");
        ensure(&weight, &weight_bytes,
               checked_elements(n, k, "weight") * element_bytes,
               "mcMalloc(weight)");
        ensure(&result, &result_bytes,
               checked_elements(m, n, "result") * element_bytes,
               "mcMalloc(result)");
    }
};

struct AttemptResult {
    bool success;
    int observed_launches;
};

AttemptResult run_attempt(mcblasHandle_t handle, Buffers* buffers,
                          int m, int n, int k, int expected,
                          MarkerBegin marker_begin, MarkerEnd marker_end,
                          MarkerError marker_error) {
    buffers->prepare(m, n, k);
    if (marker_begin(static_cast<std::size_t>(expected)) != 0) {
        std::fprintf(stderr, "cannot begin dry-launch marker: %s\n",
                     marker_error());
        std::exit(EXIT_FAILURE);
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const mcblasStatus_t status = mcblasGemmEx(
        handle, MCBLAS_OP_T, MCBLAS_OP_N, n, m, k, &alpha,
        buffers->weight, MACA_R_16BF, k, buffers->input, MACA_R_16BF, k,
        &beta, buffers->result, MACA_R_16BF, n, MCBLAS_COMPUTE_32F,
        MCBLAS_GEMM_DEFAULT);
    const int end_status = marker_end();
    check_mcblas(status, "mcblasGemmEx dry launch");
    if (end_status == 0) return {true, expected};
    int observed = 0;
    int reported_expected = 0;
    if (std::sscanf(marker_error(), "captured %d launch(es), expected %d",
                    &observed, &reported_expected) != 2
            || observed <= 0 || reported_expected != expected) {
        std::fprintf(stderr, "dry-launch marker failed: %s\n", marker_error());
        std::exit(EXIT_FAILURE);
    }
    return {false, observed};
}

void run_operation(mcblasHandle_t handle, Buffers* buffers,
                   const std::string& operation_id, int m, int n, int k,
                   int expected, const std::string& capture_path,
                   int mcblas_version, MarkerBegin marker_begin,
                   MarkerEnd marker_end, MarkerError marker_error) {
    if (setenv("MIDLAYER_C500_DRY_LAUNCH_CAPTURE",
               capture_path.c_str(), 1) != 0) {
        std::perror("setenv(MIDLAYER_C500_DRY_LAUNCH_CAPTURE)");
        std::exit(EXIT_FAILURE);
    }
    AttemptResult first = run_attempt(
        handle, buffers, m, n, k, expected,
        marker_begin, marker_end, marker_error);
    int attempts = 1;
    int stability = 1;
    int validated = expected;
    if (!first.success) {
        validated = first.observed_launches;
        const std::string first_stability = capture_path + ".stability1";
        const std::string second_stability = capture_path + ".stability2";
        if (setenv("MIDLAYER_C500_DRY_LAUNCH_CAPTURE",
                   first_stability.c_str(), 1) != 0) {
            Fail("cannot set first stability capture path");
        }
        const AttemptResult second = run_attempt(
            handle, buffers, m, n, k, validated,
            marker_begin, marker_end, marker_error);
        if (setenv("MIDLAYER_C500_DRY_LAUNCH_CAPTURE",
                   second_stability.c_str(), 1) != 0) {
            Fail("cannot set second stability capture path");
        }
        const AttemptResult third = run_attempt(
            handle, buffers, m, n, k, validated,
            marker_begin, marker_end, marker_error);
        if (!second.success || !third.success) {
            std::fprintf(stderr, "launch count failed stability validation\n");
            std::exit(EXIT_FAILURE);
        }
        attempts = 3;
        stability = 2;
    }
    if (!operation_id.empty()) {
        std::printf("%s\t%d\t%d\t%d\t%d\n", operation_id.c_str(),
                    mcblas_version, attempts, validated, stability);
        std::fflush(stdout);
    }
}

}  // namespace


int main(int argc, char** argv) {
    const auto marker_begin =
        require_interposer_symbol<MarkerBegin>("midlayer_dry_launch_begin");
    const auto marker_end =
        require_interposer_symbol<MarkerEnd>("midlayer_dry_launch_end");
    const auto marker_error =
        require_interposer_symbol<MarkerError>("midlayer_dry_launch_error");

    mcblasHandle_t handle = nullptr;
    check_mcblas(mcblasCreate(&handle), "mcblasCreate");
    int mcblas_version = 0;
    check_mcblas(
        mcblasGetVersion(handle, &mcblas_version), "mcblasGetVersion");
    Buffers buffers;
    if (argc == 1) {
        const int m = positive_environment_int("MIDLAYER_C500_GEMM_M");
        const int n = positive_environment_int("MIDLAYER_C500_GEMM_N");
        const int k = positive_environment_int("MIDLAYER_C500_GEMM_K");
        const int expected =
            positive_environment_int("MIDLAYER_C500_EXPECTED_LAUNCHES");
        const char* capture = std::getenv("MIDLAYER_C500_DRY_LAUNCH_CAPTURE");
        if (capture == nullptr || capture[0] == '\0') {
            Fail("MIDLAYER_C500_DRY_LAUNCH_CAPTURE is required");
        }
        run_operation(handle, &buffers, "", m, n, k, expected,
                      capture, mcblas_version,
                      marker_begin, marker_end, marker_error);
        std::printf("mcblas_version=%d\n", mcblas_version);
    } else if (argc == 2 && std::string(argv[1]) == "--batch") {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) Fail("batch request line must not be empty");
            std::istringstream input(line);
            std::string operation_id;
            std::string m_text;
            std::string n_text;
            std::string k_text;
            std::string expected_text;
            std::string capture_path;
            std::string extra;
            if (!(input >> operation_id >> m_text >> n_text >> k_text
                        >> expected_text >> capture_path)
                    || (input >> extra)) {
                Fail("batch request must contain six whitespace-free fields");
            }
            run_operation(
                handle, &buffers, operation_id,
                positive_int(m_text, "M"), positive_int(n_text, "N"),
                positive_int(k_text, "K"),
                positive_int(expected_text, "expected launches"),
                capture_path, mcblas_version,
                marker_begin, marker_end, marker_error);
        }
        if (!std::cin.eof()) Fail("cannot read batch requests");
    } else {
        std::fprintf(stderr, "usage: %s [--batch]\n", argv[0]);
        return 2;
    }
    check_mcblas(mcblasDestroy(handle), "mcblasDestroy");
    return EXIT_SUCCESS;
}

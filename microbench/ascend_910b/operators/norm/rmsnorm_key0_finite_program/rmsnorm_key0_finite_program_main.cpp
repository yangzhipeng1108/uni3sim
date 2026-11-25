// ACL host for the finite CANN RMSNorm key-0 memory/control program.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_rmsnorm_key0_finite_program(
    uint32_t block_dim, void* stream, void* source_x, void* source_gamma,
    void* destination_y, void* destination_rstd, void* worker_cycles,
    uint32_t rows, uint32_t hidden);

namespace {

constexpr int64_t kMaximumWorkers = 48;
constexpr int64_t kResultValuesPerWorker = 8;
constexpr int64_t kWarmupLaunches = 3;
constexpr int64_t kMeasuredLaunches = 64;
constexpr double kSystemCycleNs = 20.0;

int64_t ParseNonnegative(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int64_t ParsePositive(const char* text, const char* name) {
    const int64_t value = ParseNonnegative(text, name);
    if (value == 0) {
        std::fprintf(stderr, "%s must be positive\n", name);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t device_id = ParseNonnegative(text, "ASCEND_DEVICE_ID");
    if (device_id > 7) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be in [0, 7]\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(device_id);
}

struct Timing {
    int64_t maximum_worker_cycles;
    int64_t source_span_cycles;
};

void ValidateOutput(
        void* destination_y, void* destination_rstd,
        int64_t rows, int64_t hidden, int64_t workers) {
    uint16_t first_y = 0;
    uint16_t last_y = 0;
    float first_rstd = 0.0f;
    const size_t y_bytes = rows * hidden * sizeof(uint16_t);
    ACL_CHECK(aclrtMemcpy(
        &first_y, sizeof(first_y), destination_y, sizeof(first_y),
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        &last_y, sizeof(last_y),
        static_cast<const uint8_t*>(destination_y)
            + y_bytes - sizeof(last_y),
        sizeof(last_y), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        &first_rstd, sizeof(first_rstd), destination_rstd,
        sizeof(first_rstd), ACL_MEMCPY_DEVICE_TO_HOST));
    if (first_y != 0x3c00 || last_y != 0x3c00
            || first_rstd != 1.0f || workers <= 0) {
        std::fprintf(
            stderr,
            "output validation failed: first_y=0x%04x last_y=0x%04x "
            "first_rstd=%f\n",
            first_y, last_y, static_cast<double>(first_rstd));
        std::exit(EXIT_FAILURE);
    }
}

Timing RunOne(
        aclrtStream stream, void* source_x, void* source_gamma,
        void* destination_y, void* destination_rstd, void* worker_cycles,
        int64_t workers, int64_t rows, int64_t hidden) {
    ACL_CHECK(aclrtMemset(
        worker_cycles,
        kMaximumWorkers * kResultValuesPerWorker * sizeof(int64_t),
        0xff,
        kMaximumWorkers * kResultValuesPerWorker * sizeof(int64_t)));
    ACL_CHECK(aclrtlaunch_rmsnorm_key0_finite_program(
        static_cast<uint32_t>(workers), stream, source_x, source_gamma,
        destination_y, destination_rstd, worker_cycles,
        static_cast<uint32_t>(rows), static_cast<uint32_t>(hidden)));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> values(
        kMaximumWorkers * kResultValuesPerWorker, -1);
    ACL_CHECK(aclrtMemcpy(
        values.data(), values.size() * sizeof(int64_t), worker_cycles,
        values.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t earliest_start = -1;
    int64_t latest_end = -1;
    int64_t maximum_worker = -1;
    for (int64_t worker = 0; worker < workers; ++worker) {
        const int64_t offset = worker * kResultValuesPerWorker;
        const int64_t start = values[offset];
        const int64_t end = values[offset + 1];
        const int64_t body = values[offset + 2];
        if (start <= 0 || end <= start || body != end - start) {
            std::fprintf(stderr, "worker %ld emitted invalid cycles\n",
                         static_cast<long>(worker));
            std::exit(EXIT_FAILURE);
        }
        earliest_start = earliest_start < 0
            ? start : std::min(earliest_start, start);
        latest_end = std::max(latest_end, end);
        maximum_worker = std::max(maximum_worker, body);
    }
    return Timing{maximum_worker, latest_end - earliest_start};
}

double Median(std::vector<int64_t> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * static_cast<double>(
        values[values.size() / 2 - 1] + values[values.size() / 2]);
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <workers> <rows> <hidden>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t workers = ParsePositive(argv[1], "workers");
    const int64_t rows = ParsePositive(argv[2], "rows");
    const int64_t hidden = ParsePositive(argv[3], "hidden");
    if (workers > kMaximumWorkers || workers > rows || hidden > 8192
            || hidden % 16 != 0) {
        std::fprintf(stderr, "unsupported finite RMSNorm geometry\n");
        return EXIT_FAILURE;
    }

    const size_t x_bytes = rows * hidden * sizeof(uint16_t);
    const size_t gamma_bytes = hidden * sizeof(uint16_t);
    const size_t rstd_bytes = workers * 8 * sizeof(float);
    void* source_x = nullptr;
    void* source_gamma = nullptr;
    void* destination_y = nullptr;
    void* destination_rstd = nullptr;
    void* worker_cycles = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(&source_x, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &source_gamma, gamma_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &destination_y, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &destination_rstd, rstd_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &worker_cycles,
        kMaximumWorkers * kResultValuesPerWorker * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(source_x, x_bytes, 0x3c, x_bytes));
    ACL_CHECK(aclrtMemset(source_gamma, gamma_bytes, 0x3c, gamma_bytes));
    ACL_CHECK(aclrtMemset(destination_y, x_bytes, 0, x_bytes));
    ACL_CHECK(aclrtMemset(destination_rstd, rstd_bytes, 0, rstd_bytes));

    for (int64_t warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        RunOne(
            stream, source_x, source_gamma, destination_y, destination_rstd,
            worker_cycles, workers, rows, hidden);
    }
    std::vector<int64_t> worker_samples;
    std::vector<int64_t> span_samples;
    worker_samples.reserve(kMeasuredLaunches);
    span_samples.reserve(kMeasuredLaunches);
    for (int64_t launch = 0; launch < kMeasuredLaunches; ++launch) {
        const Timing timing = RunOne(
            stream, source_x, source_gamma, destination_y, destination_rstd,
            worker_cycles, workers, rows, hidden);
        worker_samples.push_back(timing.maximum_worker_cycles);
        span_samples.push_back(timing.source_span_cycles);
    }
    ValidateOutput(
        destination_y, destination_rstd, rows, hidden, workers);
    const double worker_cycles_median = Median(worker_samples);
    const double span_cycles_median = Median(span_samples);
    if (!(worker_cycles_median > 0.0)
            || span_cycles_median < worker_cycles_median) {
        std::fprintf(stderr, "invalid finite-program timing\n");
        return EXIT_FAILURE;
    }
    std::printf(
        "workers,rows,hidden,rows_per_full_worker,measured_launches,"
        "median_maximum_worker_cycles,median_maximum_worker_ns,"
        "median_source_span_cycles,median_source_span_ns,validation\n"
        "%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,pass\n",
        static_cast<long>(workers), static_cast<long>(rows),
        static_cast<long>(hidden),
        static_cast<long>((rows + workers - 1) / workers),
        static_cast<long>(kMeasuredLaunches), worker_cycles_median,
        worker_cycles_median * kSystemCycleNs, span_cycles_median,
        span_cycles_median * kSystemCycleNs);

    ACL_CHECK(aclrtFree(worker_cycles));
    ACL_CHECK(aclrtFree(destination_rstd));
    ACL_CHECK(aclrtFree(destination_y));
    ACL_CHECK(aclrtFree(source_gamma));
    ACL_CHECK(aclrtFree(source_x));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}

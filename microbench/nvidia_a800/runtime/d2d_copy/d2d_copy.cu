/*
 * NVIDIA A800 serial same-device D2D-copy mechanism benchmark.
 *
 * This benchmark measures one asynchronous runtime D2D copy at a time on a
 * non-default stream.  Allocation, initialization, warmup, synchronization
 * setup, and correctness validation are outside each start/copy/end event
 * interval.  It is a runtime/copy-engine mechanism benchmark, not a
 * production operator or Serving workload.
 *
 * The raw sweep is trial-first and uses fixed forward/reverse rotations so
 * every payload occupies every order position three times.  Every timed
 * interval contains exactly one copy; batched-copy amortization is forbidden.
 * The small-payload cells may establish a fixed completion floor only if
 * their raw interquartile intervals have a common intersection.  The frozen
 * D2D service term B/C_R + B/C_W is checked independently with the 64, 128,
 * and 256 MiB cells; its slope must not be refit from this capture.
 *
 * Build:
 *   /usr/local/cuda-12.2/bin/nvcc -O3 -std=c++14 -arch=sm_80 \
 *       -lineinfo -Xptxas=-v -o d2d_copy d2d_copy.cu
 *
 * Run only after proving the assigned A800 is idle:
 *   CUDA_VISIBLE_DEVICES=3 ./d2d_copy --run-id RUN_ID
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>


namespace {

constexpr std::size_t kKiB = 1024ULL;
constexpr std::size_t kMiB = 1024ULL * kKiB;
constexpr int kTrials = 48;
constexpr int kWarmupCopies = 64;
constexpr int kRequiredAsyncEngines = 1;
constexpr int kAlignmentBytes = 256;
constexpr unsigned char kSourcePattern = 0x5a;
constexpr unsigned char kDestinationPattern = 0xa5;
constexpr std::size_t kPayloads[] = {
    1ULL, 4ULL, 16ULL, 64ULL, 256ULL,
    1ULL * kKiB, 4ULL * kKiB, 16ULL * kKiB, 64ULL * kKiB,
    256ULL * kKiB, 1ULL * kMiB, 4ULL * kMiB, 16ULL * kMiB,
    64ULL * kMiB, 128ULL * kMiB, 256ULL * kMiB,
};
constexpr int kPayloadCount = sizeof(kPayloads) / sizeof(kPayloads[0]);
static_assert(kTrials == 3 * kPayloadCount,
              "three balanced rotations are required");


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


void validate_run_id(const char* run_id) {
    const std::size_t length = std::strlen(run_id);
    if (length == 0 || length > 96) {
        std::fprintf(stderr, "run_id length must be in [1,96]\n");
        std::exit(EXIT_FAILURE);
    }
    for (std::size_t index = 0; index < length; ++index) {
        const char value = run_id[index];
        const bool valid = (
            (value >= 'a' && value <= 'z')
            || (value >= 'A' && value <= 'Z')
            || (value >= '0' && value <= '9')
            || value == '.' || value == '_' || value == '-');
        if (!valid) {
            std::fprintf(
                stderr, "run_id contains unsupported character at %zu\n",
                index);
            std::exit(EXIT_FAILURE);
        }
    }
}


int payload_index_for(int trial, int order_position) {
    const int rotation = trial % kPayloadCount;
    const int cycle = trial / kPayloadCount;
    if (cycle == 1) {
        return (
            rotation - order_position + kPayloadCount) % kPayloadCount;
    }
    if (cycle == 0 || cycle == 2) {
        return (rotation + order_position) % kPayloadCount;
    }
    std::fprintf(stderr, "trial schedule exceeds declared rotations\n");
    std::exit(EXIT_FAILURE);
}


void validate_schedule() {
    int position_counts[kPayloadCount][kPayloadCount] = {};
    for (int trial = 0; trial < kTrials; ++trial) {
        bool seen[kPayloadCount] = {};
        for (int position = 0; position < kPayloadCount; ++position) {
            const int payload = payload_index_for(trial, position);
            if (payload < 0 || payload >= kPayloadCount || seen[payload]) {
                std::fprintf(
                    stderr, "invalid payload order trial=%d position=%d\n",
                    trial, position);
                std::exit(EXIT_FAILURE);
            }
            seen[payload] = true;
            ++position_counts[position][payload];
        }
    }
    for (int position = 0; position < kPayloadCount; ++position) {
        for (int payload = 0; payload < kPayloadCount; ++payload) {
            if (position_counts[position][payload] != 3) {
                std::fprintf(
                    stderr,
                    "unbalanced payload position=%d payload=%d count=%d\n",
                    position, payload, position_counts[position][payload]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
}


void print_uuid(const cudaUUID_t& uuid) {
    for (int index = 0; index < 16; ++index) {
        std::fprintf(
            stderr, "%02x",
            static_cast<unsigned char>(uuid.bytes[index]));
    }
}


void validate_sample(
        const unsigned char* source, const unsigned char* destination,
        std::size_t offset) {
    unsigned char source_value = 0;
    unsigned char destination_value = 0;
    CHECK_CUDA(cudaMemcpy(
        &source_value, source + offset, 1, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        &destination_value, destination + offset, 1,
        cudaMemcpyDeviceToHost));
    if (source_value != kSourcePattern
            || destination_value != kSourcePattern) {
        std::fprintf(
            stderr,
            "validation failed at offset=%zu source=%u destination=%u\n",
            offset, static_cast<unsigned int>(source_value),
            static_cast<unsigned int>(destination_value));
        std::exit(EXIT_FAILURE);
    }
}


struct Observation {
    int trial;
    int order_position;
    std::size_t payload_bytes;
    double event_ns;
};

}  // namespace


int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--run-id") != 0) {
        std::fprintf(stderr, "usage: %s --run-id RUN_ID\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* run_id = argv[2];
    validate_run_id(run_id);
    validate_schedule();

    int device = -1;
    CHECK_CUDA(cudaGetDevice(&device));
    if (device != 0) {
        std::fprintf(
            stderr,
            "CUDA-visible device must be index 0 after isolation, got %d\n",
            device);
        return EXIT_FAILURE;
    }
    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108
            || properties.asyncEngineCount < kRequiredAsyncEngines) {
        std::fprintf(
            stderr,
            "unexpected device cc=%d.%d sms=%d async_engines=%d\n",
            properties.major, properties.minor,
            properties.multiProcessorCount, properties.asyncEngineCount);
        return EXIT_FAILURE;
    }
    int driver_version = 0;
    int runtime_version = 0;
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    char pci_bus_id[32] = {};
    CHECK_CUDA(cudaDeviceGetPCIBusId(
        pci_bus_id, sizeof(pci_bus_id), device));
    std::fprintf(
        stderr,
        "run_id=%s device=%s pci=%s uuid=GPU-",
        run_id, properties.name, pci_bus_id);
    print_uuid(properties.uuid);
    std::fprintf(
        stderr,
        " cc=%d.%d sms=%d async_engines=%d l2_bytes=%d "
        "clock_khz=%d memory_clock_khz=%d driver=%d runtime=%d "
        "trials=%d warmup_copies=%d maximum_payload_bytes=%zu\n",
        properties.major, properties.minor,
        properties.multiProcessorCount, properties.asyncEngineCount,
        properties.l2CacheSize, properties.clockRate,
        properties.memoryClockRate, driver_version, runtime_version,
        kTrials, kWarmupCopies, kPayloads[kPayloadCount - 1]);

    unsigned char* source = nullptr;
    unsigned char* destination = nullptr;
    const std::size_t maximum_payload = kPayloads[kPayloadCount - 1];
    CHECK_CUDA(cudaMalloc(&source, maximum_payload));
    CHECK_CUDA(cudaMalloc(&destination, maximum_payload));
    if (reinterpret_cast<std::uintptr_t>(source) % kAlignmentBytes != 0
            || reinterpret_cast<std::uintptr_t>(destination)
                % kAlignmentBytes != 0
            || source == destination) {
        std::fprintf(stderr, "invalid D2D allocation identity/alignment\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemset(source, kSourcePattern, maximum_payload));
    CHECK_CUDA(cudaMemset(destination, kDestinationPattern, maximum_payload));

    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t end;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    for (int index = 0; index < kWarmupCopies; ++index) {
        CHECK_CUDA(cudaMemcpyAsync(
            destination, source, maximum_payload,
            cudaMemcpyDeviceToDevice, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<Observation> observations;
    observations.reserve(
        static_cast<std::size_t>(kTrials) * kPayloadCount);
    for (int trial = 0; trial < kTrials; ++trial) {
        for (int position = 0; position < kPayloadCount; ++position) {
            const int payload_index = payload_index_for(trial, position);
            const std::size_t payload = kPayloads[payload_index];
            CHECK_CUDA(cudaEventRecord(start, stream));
            CHECK_CUDA(cudaMemcpyAsync(
                destination, source, payload,
                cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaEventRecord(end, stream));
            CHECK_CUDA(cudaEventSynchronize(end));
            float elapsed_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, end));
            const double event_ns = static_cast<double>(elapsed_ms) * 1.0e6;
            if (!std::isfinite(event_ns) || event_ns <= 0.0) {
                std::fprintf(
                    stderr,
                    "invalid event duration trial=%d position=%d ns=%.9f\n",
                    trial, position, event_ns);
                return EXIT_FAILURE;
            }
            observations.push_back(Observation{
                trial, position, payload, event_ns});
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    validate_sample(source, destination, 0);
    validate_sample(source, destination, maximum_payload / 2);
    validate_sample(source, destination, maximum_payload - 1);

    std::printf(
        "run_id,trial,order_position,payload_bytes,source_role,"
        "destination_role,event_ns,validation\n");
    for (const Observation& observation : observations) {
        std::printf(
            "%s,%d,%d,%zu,A,B,%.3f,pass\n",
            run_id, observation.trial, observation.order_position,
            observation.payload_bytes, observation.event_ns);
    }
    std::fflush(stdout);
    std::fprintf(stderr, "validation=pass observations=%zu\n",
                 observations.size());

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(source));
    CHECK_CUDA(cudaFree(destination));
    return EXIT_SUCCESS;
}

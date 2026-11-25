// ACL host for the independent mixed-core AIV workspace-transfer benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aiv_workspace_transfer_contract.h"

#define ACL_CHECK(call) do {                                               \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_workspace_transfer(
    uint32_t block_dim, void* stream, void* vec1_source,
    void* vec1_destination, void* vec2_state, void* cycles,
    void* validation, int64_t mode, int64_t iterations,
    int64_t validation_only);

namespace {

using namespace aiv_workspace_transfer_contract;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr || *text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t value = ParseInteger(text, "ASCEND_DEVICE_ID");
    if (value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID is outside int32 range\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

const char* ModeName(Mode mode) {
    switch (mode) {
        case Mode::kVec1Load4096Isolated:
            return "vec1_load_4096_isolated";
        case Mode::kVec1Store2048Isolated:
            return "vec1_store_2048_isolated";
        case Mode::kVec1OrderedChain:
            return "vec1_ordered_chain";
        case Mode::kVec2Load256Isolated:
            return "vec2_load_256_isolated";
        case Mode::kVec2Store256Isolated:
            return "vec2_store_256_isolated";
        case Mode::kVec2OrderedChain:
            return "vec2_ordered_chain";
        case Mode::kVec1M4S512OrderedChain:
            return "vec1_m4_s512_ordered_chain";
        case Mode::kVec1M4S1024OrderedChain:
            return "vec1_m4_s1024_ordered_chain";
        case Mode::kVec2M4FinalOrderedChain:
            return "vec2_m4_final_ordered_chain";
        default:
            std::fprintf(stderr, "unsupported mode enum\n");
            std::exit(EXIT_FAILURE);
    }
}

struct ActorCyclePair {
    int64_t actor_id;
    int64_t short_cycles;
    int64_t long_cycles;
};

std::string JoinPairs(const std::vector<ActorCyclePair>& pairs) {
    if (pairs.empty()) {
        return "none";
    }
    std::string output;
    for (size_t index = 0; index < pairs.size(); ++index) {
        if (index != 0) {
            output += ";";
        }
        output += std::to_string(pairs[index].actor_id);
        output += ":";
        output += std::to_string(pairs[index].short_cycles);
        output += ":";
        output += std::to_string(pairs[index].long_cycles);
    }
    return output;
}

struct Cell {
    Mode mode;
    int64_t aic_groups;
    int64_t iterations;
};

struct DomainCycles {
    std::vector<ActorCyclePair> vec1;
    std::vector<ActorCyclePair> vec2;
};

DomainCycles RunPaired(
        aclrtStream stream, void* vec1_source, void* vec1_destination,
        void* vec2_state, void* cycles, void* validation,
        const Cell& cell, int64_t iterations) {
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kCycleSlotElements * sizeof(int64_t));
    const size_t validation_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kValidationSlotElements * sizeof(float));
    ACL_CHECK(aclrtMemset(cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtMemset(validation, validation_bytes, 0, validation_bytes));
    ACL_CHECK(aclrtlaunch_aiv_workspace_transfer(
        static_cast<uint32_t>(cell.aic_groups), stream, vec1_source,
        vec1_destination, vec2_state, cycles, validation,
        static_cast<int64_t>(cell.mode), iterations, 0));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> host_cycles(static_cast<size_t>(
        kMaximumAivWorkers * kCycleSlotElements), 0);
    std::vector<float> host_validation(static_cast<size_t>(
        kMaximumAivWorkers * kValidationSlotElements), 0.0f);
    ACL_CHECK(aclrtMemcpy(
        host_cycles.data(), cycle_bytes, cycles, cycle_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        host_validation.data(), validation_bytes, validation,
        validation_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    DomainCycles result;
    result.vec1.reserve(static_cast<size_t>(cell.aic_groups));
    result.vec2.reserve(static_cast<size_t>(cell.aic_groups));
    for (int64_t worker = 0; worker < cell.aic_groups * kAivPerAic;
         ++worker) {
        const bool primary = worker % kAivPerAic == 0;
        const bool vec1 = UsesVec1(cell.mode);
        const bool active = primary;
        const size_t cycle_slot = static_cast<size_t>(
            worker * kCycleSlotElements);
        const int64_t short_cycles = host_cycles[cycle_slot];
        const int64_t long_cycles = host_cycles[cycle_slot + 1];
        const float observed = host_validation[static_cast<size_t>(
            worker * kValidationSlotElements)];
        if (active) {
            if (short_cycles <= 0 || long_cycles <= short_cycles
                    || !std::isfinite(observed)
                    || std::fabs(observed - 1.0f) > 1.0e-3f) {
                std::fprintf(
                    stderr,
                    "active AIV %ld paired validation failed: "
                    "short=%ld long=%ld value=%g\n",
                    static_cast<long>(worker),
                    static_cast<long>(short_cycles),
                    static_cast<long>(long_cycles), observed);
                std::exit(EXIT_FAILURE);
            }
            (vec1 ? result.vec1 : result.vec2).push_back(ActorCyclePair{
                worker, short_cycles, long_cycles});
        } else if (short_cycles != 0 || long_cycles != 0
                   || observed != 0.0f) {
            std::fprintf(stderr, "inactive AIV %ld executed work\n",
                         static_cast<long>(worker));
            std::exit(EXIT_FAILURE);
        }
    }
    return result;
}

void ValidateCell(
        aclrtStream stream, void* vec1_source, void* vec1_destination,
        void* vec2_state, void* cycles, void* validation,
        const Cell& cell) {
    const size_t actor_half_elements = static_cast<size_t>(
        kMaximumAivWorkers * kActorStrideBytes / sizeof(uint16_t));
    const size_t actor_float_elements = static_cast<size_t>(
        kMaximumAivWorkers * kActorStrideBytes / sizeof(float));
    const size_t actor_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kActorStrideBytes);
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kCycleSlotElements * sizeof(int64_t));
    const size_t validation_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kValidationSlotElements * sizeof(float));
    const uint16_t source_bits = 0x4000;
    const uint16_t destination_bits = 0xc200;
    std::vector<uint16_t> source(actor_half_elements, source_bits);
    std::vector<uint16_t> destination(actor_half_elements, destination_bits);
    std::vector<float> state(actor_float_elements, kVec2StateSentinel);
    ACL_CHECK(aclrtMemcpy(vec1_source, actor_bytes, source.data(), actor_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vec1_destination, actor_bytes, destination.data(),
                         actor_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vec2_state, actor_bytes, state.data(), actor_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtMemset(validation, validation_bytes, 0, validation_bytes));
    ACL_CHECK(aclrtlaunch_aiv_workspace_transfer(
        static_cast<uint32_t>(cell.aic_groups), stream, vec1_source,
        vec1_destination, vec2_state, cycles, validation,
        static_cast<int64_t>(cell.mode), 1, 1));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<float> validation_values(static_cast<size_t>(
        kMaximumAivWorkers * kValidationSlotElements), 0.0f);
    ACL_CHECK(aclrtMemcpy(source.data(), actor_bytes, vec1_source, actor_bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(destination.data(), actor_bytes, vec1_destination,
                         actor_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(state.data(), actor_bytes, vec2_state, actor_bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(validation_values.data(), validation_bytes,
                         validation, validation_bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t index = 0; index < source.size(); ++index) {
        if (source[index] != source_bits) {
            std::fprintf(stderr, "Vec1 source or guard changed at %zu\n", index);
            std::exit(EXIT_FAILURE);
        }
    }
    const size_t half_stride = static_cast<size_t>(
        kActorStrideBytes / sizeof(uint16_t));
    const size_t float_stride = static_cast<size_t>(
        kActorStrideBytes / sizeof(float));
    size_t vec1_store_elements = static_cast<size_t>(
        kVec1StoreBytes / sizeof(uint16_t));
    if (cell.mode == Mode::kVec1M4S512OrderedChain) {
        vec1_store_elements = static_cast<size_t>(
            kVec1M4S512StoreBytes / sizeof(uint16_t));
    } else if (cell.mode == Mode::kVec1M4S1024OrderedChain) {
        vec1_store_elements = static_cast<size_t>(
            kVec1M4S1024StoreBytes / sizeof(uint16_t));
    }
    const size_t vec2_state_elements = static_cast<size_t>(
        (cell.mode == Mode::kVec2M4FinalOrderedChain
         ? kVec2M4FinalStoreBytes : kVec2StateBytes) / sizeof(float));
    for (int64_t worker = 0; worker < kMaximumAivWorkers; ++worker) {
        const bool launched = worker < cell.aic_groups * kAivPerAic;
        const bool active = launched && worker % kAivPerAic == 0;
        const bool vec1_store = active && (
            cell.mode == Mode::kVec1Store2048Isolated
            || cell.mode == Mode::kVec1OrderedChain
            || cell.mode == Mode::kVec1M4S512OrderedChain
            || cell.mode == Mode::kVec1M4S1024OrderedChain);
        const uint16_t expected_vec1 = UsesVec1(cell.mode)
            ? source_bits : 0x3c00;
        for (size_t element = 0; element < half_stride; ++element) {
            const uint16_t expected = vec1_store && element < vec1_store_elements
                ? expected_vec1 : destination_bits;
            if (destination[static_cast<size_t>(worker) * half_stride + element]
                    != expected) {
                std::fprintf(stderr,
                             "Vec1 destination/guard mismatch worker=%ld element=%zu\n",
                             static_cast<long>(worker), element);
                std::exit(EXIT_FAILURE);
            }
        }
        const bool vec2_store = active && (
            cell.mode == Mode::kVec2Store256Isolated
            || IsVec2Ordered(cell.mode));
        const float expected_vec2 = IsVec2Ordered(cell.mode)
            ? kVec2ChainStoreSentinel : kLocalSentinel;
        for (size_t element = 0; element < float_stride; ++element) {
            const float expected = vec2_store && element < vec2_state_elements
                ? expected_vec2 : kVec2StateSentinel;
            if (state[static_cast<size_t>(worker) * float_stride + element]
                    != expected) {
                std::fprintf(stderr,
                             "Vec2 state/guard mismatch worker=%ld element=%zu\n",
                             static_cast<long>(worker), element);
                std::exit(EXIT_FAILURE);
            }
        }
        const size_t slot = static_cast<size_t>(
            worker * kValidationSlotElements);
        if (!active) {
            if (validation_values[slot] != 0.0f
                    || validation_values[slot + 1] != 0.0f) {
                std::fprintf(stderr, "inactive AIV validation changed\n");
                std::exit(EXIT_FAILURE);
            }
            continue;
        }
        float expected_loaded = kLocalSentinel;
        if (cell.mode == Mode::kVec1Load4096Isolated
                || UsesVec1(cell.mode)) {
            expected_loaded = kVec1SourceSentinel;
        } else if (cell.mode == Mode::kVec2Load256Isolated
                   || IsVec2Ordered(cell.mode)) {
            expected_loaded = kVec2StateSentinel;
        }
        const float expected_final = IsVec2Ordered(cell.mode)
            ? kVec2ChainStoreSentinel : expected_loaded;
        if (validation_values[slot] != expected_loaded
                || validation_values[slot + 1] != expected_final) {
            std::fprintf(stderr, "active AIV sentinel validation failed\n");
            std::exit(EXIT_FAILURE);
        }
    }
    // Sentinel validation is deliberately outside the timed interval.  Put
    // every input back to the original measurement state before Run reads it.
    std::fill(source.begin(), source.end(), static_cast<uint16_t>(0x3c00));
    std::fill(destination.begin(), destination.end(), static_cast<uint16_t>(0));
    std::fill(state.begin(), state.end(), kLocalSentinel);
    ACL_CHECK(aclrtMemcpy(vec1_source, actor_bytes, source.data(), actor_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vec1_destination, actor_bytes, destination.data(),
                         actor_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vec2_state, actor_bytes, state.data(), actor_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
}

double CriticalPairedDifferentialNs(
        const std::vector<ActorCyclePair>& pairs, int64_t iterations) {
    if (pairs.empty()) {
        return 0.0;
    }
    double critical_ns = 0.0;
    for (const ActorCyclePair& pair : pairs) {
        if (pair.long_cycles <= pair.short_cycles) {
            std::fprintf(
                stderr, "actor %ld long interval must exceed short interval\n",
                static_cast<long>(pair.actor_id));
            std::exit(EXIT_FAILURE);
        }
        const double actor_ns = static_cast<double>(
            pair.long_cycles - pair.short_cycles)
            * kSystemCounterTickNs / static_cast<double>(3 * iterations);
        critical_ns = std::max(critical_ns, actor_ns);
    }
    return critical_ns;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace aiv_workspace_transfer_contract;
    if (argc < 5 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 3 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch MODE AIC_GROUPS ITERATIONS [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    for (int offset = 2; offset < argc; offset += 3) {
        const int64_t mode_value = ParseInteger(argv[offset], "mode");
        const int64_t groups = ParseInteger(argv[offset + 1], "aic_groups");
        const int64_t iterations = ParseInteger(
            argv[offset + 2], "iterations");
        if (!IsSupportedMode(mode_value) || !IsSupportedGroupCount(groups)
                || iterations <= 0 || iterations > INT32_MAX / 4) {
            std::fprintf(stderr, "invalid workspace-transfer cell\n");
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{
            static_cast<Mode>(mode_value), groups, iterations});
    }

    const size_t actor_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kActorStrideBytes);
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kCycleSlotElements * sizeof(int64_t));
    const size_t validation_bytes = static_cast<size_t>(
        kMaximumAivWorkers * kValidationSlotElements * sizeof(float));
    std::vector<uint16_t> host_vec1(actor_bytes / sizeof(uint16_t), 0x3c00);
    std::vector<float> host_vec2(actor_bytes / sizeof(float), 1.0f);
    void* vec1_source = nullptr;
    void* vec1_destination = nullptr;
    void* vec2_state = nullptr;
    void* cycles = nullptr;
    void* validation = nullptr;
    aclrtStream stream = nullptr;
    const int device_id = DeviceId();
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &vec1_source, actor_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &vec1_destination, actor_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &vec2_state, actor_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &validation, validation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        vec1_source, actor_bytes, host_vec1.data(), actor_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(
        vec1_destination, actor_bytes, 0, actor_bytes));
    ACL_CHECK(aclrtMemcpy(
        vec2_state, actor_bytes, host_vec2.data(), actor_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    std::puts(
        "mode,aic_groups,aiv_workers,active_vec1_workers,"
        "active_vec2_workers,iterations,vec1_actor_cycle_pairs,"
        "vec2_actor_cycle_pairs,tick_ns,"
        "vec1_critical_completion_ns,vec2_critical_completion_ns,"
        "critical_completion_ns,"
        "validation");
    for (const Cell& cell : cells) {
        ValidateCell(stream, vec1_source, vec1_destination, vec2_state,
                     cycles, validation, cell);
        const DomainCycles paired_cycles = RunPaired(
            stream, vec1_source, vec1_destination, vec2_state,
            cycles, validation, cell, cell.iterations);
        const double vec1_ns = CriticalPairedDifferentialNs(
            paired_cycles.vec1, cell.iterations);
        const double vec2_ns = CriticalPairedDifferentialNs(
            paired_cycles.vec2, cell.iterations);
        const double critical_ns = std::max(vec1_ns, vec2_ns);
        if (!std::isfinite(critical_ns) || critical_ns <= 0.0) {
            std::fprintf(stderr, "derived completion time is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%s,%ld,%ld,%zu,%zu,%ld,%s,%s,%.1f,%.12f,%.12f,"
            "%.12f,pass\n",
            ModeName(cell.mode), static_cast<long>(cell.aic_groups),
            static_cast<long>(cell.aic_groups * kAivPerAic),
            paired_cycles.vec1.size(), paired_cycles.vec2.size(),
            static_cast<long>(cell.iterations),
            JoinPairs(paired_cycles.vec1).c_str(),
            JoinPairs(paired_cycles.vec2).c_str(), kSystemCounterTickNs,
            vec1_ns, vec2_ns, critical_ns);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(validation));
    ACL_CHECK(aclrtFree(cycles));
    ACL_CHECK(aclrtFree(vec2_state));
    ACL_CHECK(aclrtFree(vec1_destination));
    ACL_CHECK(aclrtFree(vec1_source));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}

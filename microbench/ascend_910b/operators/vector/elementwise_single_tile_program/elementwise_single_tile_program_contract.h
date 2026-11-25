#pragma once

#include <cstdint>

namespace elementwise_single_tile_program_contract {

constexpr int64_t kAivWorkers = 47;
constexpr int64_t kFullTileElements = 1408;
constexpr int64_t kTailTileElements = 768;
constexpr int64_t kMaximumDtypeBytes = 4;
constexpr int64_t kDagBufferCount = 5;
constexpr int64_t kTbufBytes =
    kFullTileElements * kMaximumDtypeBytes * kDagBufferCount;
constexpr int64_t kWarmupLaunches = 3;
constexpr int64_t kMeasuredLaunches = 64;
constexpr int64_t kResultValuesPerWorker = 8;
constexpr double kSystemCycleNs = 20.0;

}  // namespace elementwise_single_tile_program_contract

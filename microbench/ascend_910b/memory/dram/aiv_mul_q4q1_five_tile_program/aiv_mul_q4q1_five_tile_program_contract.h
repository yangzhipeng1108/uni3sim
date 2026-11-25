#pragma once

#include <cstdint>

namespace aiv_mul_q4q1_five_tile_program_contract {

constexpr int64_t kAivWorkers = 48;
constexpr int64_t kFullWorkers = 47;
constexpr int64_t kFullTileElements = 8832;
constexpr int64_t kFullTilesPerWorker = 4;
constexpr int64_t kFullWorkerTailElements = 8448;
constexpr int64_t kLastWorkerTailElements = 4352;
constexpr int64_t kBlockFactorElements = 43776;
constexpr int64_t kLastWorkerElements = 39680;
constexpr int64_t kTotalElements = 2097152;
constexpr int64_t kWarmupLaunches = 3;
constexpr int64_t kMeasuredLaunches = 64;
constexpr uint16_t kOutputHalfOne = 0x3c00;
constexpr double kSystemCycleNs = 20.0;

static_assert(
    kFullTilesPerWorker * kFullTileElements
        + kFullWorkerTailElements == kBlockFactorElements);
static_assert(
    kFullTilesPerWorker * kFullTileElements
        + kLastWorkerTailElements == kLastWorkerElements);
static_assert(
    kFullWorkers * kBlockFactorElements
        + kLastWorkerElements == kTotalElements);

}  // namespace aiv_mul_q4q1_five_tile_program_contract

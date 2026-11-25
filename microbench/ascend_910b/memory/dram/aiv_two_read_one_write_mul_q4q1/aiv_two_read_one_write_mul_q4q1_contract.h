#pragma once

#include <cstdint>

namespace aiv_two_read_one_write_mul_q4q1_contract {

constexpr int64_t kAivWorkers = 48;
constexpr int64_t kWarmupLaunches = 3;
constexpr int64_t kMeasuredLaunches = 64;
constexpr int64_t kMaximumTilesPerWorker = 4;
constexpr int64_t kMinimumTileBytes = 2048;
constexpr int64_t kMaximumTileBytes = 19456;
constexpr int64_t kAlignmentBytes = 32;
constexpr uint16_t kOutputHalfOne = 0x3c00;
constexpr double kSystemCycleNs = 20.0;

inline bool IsSupportedTileBytes(int64_t tile_bytes) {
    return tile_bytes >= kMinimumTileBytes
        && tile_bytes <= kMaximumTileBytes
        && tile_bytes % kAlignmentBytes == 0;
}

inline bool IsSupportedTilesPerWorker(int64_t tiles_per_worker) {
    return tiles_per_worker >= 1
        && tiles_per_worker <= kMaximumTilesPerWorker;
}

}  // namespace aiv_two_read_one_write_mul_q4q1_contract

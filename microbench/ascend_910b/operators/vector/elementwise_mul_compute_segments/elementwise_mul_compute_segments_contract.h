#pragma once

#include <cstdint>

namespace elementwise_mul_compute_segments_contract {

constexpr int64_t kPrefixCastA = 0;
constexpr int64_t kSuffixCastBMulCastOut = 1;
constexpr int64_t kMaximumWorkers = 48;
constexpr int64_t kMaximumElements = 8832;
constexpr int64_t kResultValuesPerWorker = 8;
constexpr double kSystemCycleNs = 20.0;

}  // namespace elementwise_mul_compute_segments_contract

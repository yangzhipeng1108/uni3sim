#pragma once

#include <cstdint>

namespace cross_domain_contract {

constexpr int64_t kMaximumAicBlocks = 24;
constexpr int64_t kAivPerAic = 2;
constexpr int64_t kActorsPerAicGroup = 1 + kAivPerAic;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));
constexpr int64_t kMaximumActors =
    kMaximumAicBlocks * kActorsPerAicGroup;

constexpr int64_t kTaskCounts[] = {1, 2, 4, 8, 16, 32, 64, 96, 128};
constexpr int64_t kTaskCountLength =
    static_cast<int64_t>(sizeof(kTaskCounts) / sizeof(kTaskCounts[0]));

#ifdef CROSS_DOMAIN_CONTRACT_AICORE
__aicore__ inline bool IsFrozenTaskCount(int64_t value) {
#else
inline bool IsFrozenTaskCount(int64_t value) {
#endif
    for (int64_t index = 0; index < kTaskCountLength; ++index) {
        if (kTaskCounts[index] == value) {
            return true;
        }
    }
    return false;
}

}  // namespace cross_domain_contract

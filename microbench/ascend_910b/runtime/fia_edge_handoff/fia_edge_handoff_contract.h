#pragma once

#include <cstdint>

namespace fia_edge_handoff_contract {

constexpr int64_t kMaximumAicGroups = 24;
constexpr int64_t kAivPerAic = 2;
constexpr int64_t kActorsPerAicGroup = 3;
constexpr int64_t kResultSlotBytes = 128;
constexpr int64_t kResultSlotElements =
    kResultSlotBytes / static_cast<int64_t>(sizeof(int64_t));
constexpr int64_t kMaximumActors =
    kMaximumAicGroups * kActorsPerAicGroup;
constexpr int64_t kTaskCounts[] = {4, 8, 16, 32, 64, 92, 128};
constexpr int64_t kTaskCountLength =
    static_cast<int64_t>(sizeof(kTaskCounts) / sizeof(kTaskCounts[0]));

#ifdef FIA_EDGE_HANDOFF_CONTRACT_AICORE
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

}  // namespace fia_edge_handoff_contract

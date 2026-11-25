#pragma once

#include <cstdint>

namespace fia_program_span_contract {

constexpr int64_t kMaximumAicGroups = 24;
constexpr int64_t kAivPerAic = 2;
constexpr int64_t kActorsPerAicGroup = 1 + kAivPerAic;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));
constexpr int64_t kMaximumActors =
    kMaximumAicGroups * kActorsPerAicGroup;
constexpr int64_t kResultElements = 2;
constexpr int64_t kProgramRepetitions = 8;
constexpr int64_t kMinimumTaskCount = 1;
constexpr int64_t kMaximumTaskCount = 12;

inline bool IsFrozenAicGroupCount(int64_t value) {
    return value == 1 || value == 8 || value == 16
        || value == 22 || value == 24;
}

#ifdef FIA_PROGRAM_SPAN_CONTRACT_AICORE
__aicore__ inline bool IsFrozenTaskCount(int64_t value) {
#else
inline bool IsFrozenTaskCount(int64_t value) {
#endif
    return value >= kMinimumTaskCount && value <= kMaximumTaskCount;
}

}  // namespace fia_program_span_contract

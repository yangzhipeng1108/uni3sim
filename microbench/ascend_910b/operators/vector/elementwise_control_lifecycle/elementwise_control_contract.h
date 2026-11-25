#pragma once

#include <cstdint>

namespace elementwise_control_contract {

constexpr int64_t kAivWorkers = 48;
constexpr int64_t kResultSlotBytes = 64;
constexpr int64_t kResultSlotElements =
    kResultSlotBytes / static_cast<int64_t>(sizeof(int64_t));
constexpr int64_t kLogicalBufferCount = 6;
constexpr int64_t kPingPongOffset = 6;
constexpr int64_t kBufferCount = 12;
constexpr int64_t kBufferBytes = 32;
constexpr int64_t kPoolBytes = kBufferCount * kBufferBytes;
constexpr int64_t kDagNodeCount = 7;
constexpr int64_t kBufferLookupsPerTile = 12;
constexpr int64_t kMte2LocksPerTile = 2;
constexpr int64_t kVectorLocksPerTile = 9;
constexpr int64_t kMte3LocksPerTile = 1;

enum class Variant : int64_t {
    kControlBaseline = 0,
    kDagTbuf = 1,
    kFullLifecycle = 2,
};

#ifdef ELEMENTWISE_CONTROL_CONTRACT_AICORE
__aicore__ inline bool IsFrozenTileCount(int64_t value) {
#else
inline bool IsFrozenTileCount(int64_t value) {
#endif
    return value == 18 || value == 29 || value == 30
        || value == 36 || value == 58 || value == 59;
}

#ifdef ELEMENTWISE_CONTROL_CONTRACT_AICORE
__aicore__ inline bool IsFrozenVariant(int64_t value) {
#else
inline bool IsFrozenVariant(int64_t value) {
#endif
    return value >= static_cast<int64_t>(Variant::kControlBaseline)
        && value <= static_cast<int64_t>(Variant::kFullLifecycle);
}

}  // namespace elementwise_control_contract

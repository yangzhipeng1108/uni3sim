#ifndef MIDLAYER_ASCEND_910B_AIV_WORKSPACE_TRANSFER_CONTRACT_H_
#define MIDLAYER_ASCEND_910B_AIV_WORKSPACE_TRANSFER_CONTRACT_H_

#include <cstdint>

namespace aiv_workspace_transfer_contract {

constexpr int64_t kAivPerAic = 2;
constexpr int64_t kMaximumAicGroups = 24;
constexpr int64_t kMaximumAivWorkers = kAivPerAic * kMaximumAicGroups;
constexpr int64_t kCycleSlotElements = 8;
constexpr int64_t kValidationSlotElements = 16;
constexpr int64_t kActorStrideBytes = 32768;
constexpr int64_t kVec1LoadBytes = 4096;
constexpr int64_t kVec1StoreBytes = 2048;
constexpr int64_t kVec2StateBytes = 256;
constexpr int64_t kVec1M4S512LoadBytes = 8192;
constexpr int64_t kVec1M4S512StoreBytes = 4096;
constexpr int64_t kVec1M4S1024LoadBytes = 16384;
constexpr int64_t kVec1M4S1024StoreBytes = 8192;
constexpr int64_t kVec2M4FinalReadBytes = 1024;
constexpr int64_t kVec2M4FinalStoreBytes = 512;
constexpr int64_t kWarmupIterations = 16;
constexpr double kSystemCounterTickNs = 20.0;
constexpr float kLocalSentinel = 1.0f;
constexpr float kVec1SourceSentinel = 2.0f;
constexpr float kVec1DestinationSentinel = -3.0f;
constexpr float kVec2StateSentinel = 3.0f;
constexpr float kVec2ChainStoreSentinel = 4.0f;

enum class Mode : int64_t {
    kVec1Load4096Isolated = 0,
    kVec1Store2048Isolated = 1,
    kVec1OrderedChain = 2,
    kVec2Load256Isolated = 3,
    kVec2Store256Isolated = 4,
    kVec2OrderedChain = 5,
    kVec1M4S512OrderedChain = 6,
    kVec1M4S1024OrderedChain = 7,
    kVec2M4FinalOrderedChain = 8,
};

#ifdef AIV_WORKSPACE_TRANSFER_CONTRACT_AICORE
#define AIV_WORKSPACE_TRANSFER_INLINE __aicore__ inline
#else
#define AIV_WORKSPACE_TRANSFER_INLINE inline
#endif

AIV_WORKSPACE_TRANSFER_INLINE bool IsSupportedGroupCount(int64_t groups) {
    return groups == 1 || groups == 8 || groups == 16 || groups == 24;
}

AIV_WORKSPACE_TRANSFER_INLINE bool IsSupportedMode(int64_t mode) {
    return mode >= static_cast<int64_t>(Mode::kVec1Load4096Isolated)
        && mode <= static_cast<int64_t>(Mode::kVec2M4FinalOrderedChain);
}

AIV_WORKSPACE_TRANSFER_INLINE bool UsesVec1(Mode mode) {
    return mode == Mode::kVec1Load4096Isolated
        || mode == Mode::kVec1Store2048Isolated
        || mode == Mode::kVec1OrderedChain
        || mode == Mode::kVec1M4S512OrderedChain
        || mode == Mode::kVec1M4S1024OrderedChain;
}

AIV_WORKSPACE_TRANSFER_INLINE bool IsVec2Ordered(Mode mode) {
    return mode == Mode::kVec2OrderedChain
        || mode == Mode::kVec2M4FinalOrderedChain;
}

AIV_WORKSPACE_TRANSFER_INLINE bool UsesVec2(Mode mode) {
    return mode == Mode::kVec2Load256Isolated
        || mode == Mode::kVec2Store256Isolated
        || mode == Mode::kVec2OrderedChain;
}

#undef AIV_WORKSPACE_TRANSFER_INLINE

}  // namespace aiv_workspace_transfer_contract

#endif  // MIDLAYER_ASCEND_910B_AIV_WORKSPACE_TRANSFER_CONTRACT_H_

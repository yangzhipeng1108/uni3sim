#ifndef AIC_MTE2_GEMM_REGION_SCHEDULE_H
#define AIC_MTE2_GEMM_REGION_SCHEDULE_H

#include <cstdint>

namespace GemmRegionSchedule {

constexpr int64_t kActiveAics = 24;

struct TileIndex {
    int64_t m;
    int64_t n;
};

__aicore__ inline int64_t GreatestCommonDivisor(
        int64_t left, int64_t right) {
    while (right != 0) {
        const int64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

__aicore__ inline int64_t WorkerRounds(
        int64_t block, int64_t region, int64_t m_block,
        int64_t n_block) {
    const int64_t tiles_per_region = m_block * n_block;
    const int64_t tail_aics_per_region =
        tiles_per_region % kActiveAics;
    const int64_t rounds =
        (tiles_per_region + kActiveAics - 1) / kActiveAics;
    const int64_t block_start =
        region * tail_aics_per_region % kActiveAics;
    const int64_t relative_block =
        (block + kActiveAics - block_start) % kActiveAics;
    return tail_aics_per_region == 0
        || relative_block < tail_aics_per_region ? rounds : rounds - 1;
}

__aicore__ inline TileIndex WorkerTile(
        int64_t block, int64_t region, int64_t round,
        int64_t n_regions, int64_t m_block, int64_t n_block) {
    const int64_t tail_aics_per_region =
        m_block * n_block % kActiveAics;
    const int64_t block_start =
        region * tail_aics_per_region % kActiveAics;
    const int64_t local_block = (
        (block + kActiveAics - block_start) % kActiveAics
        + round * kActiveAics);
    const int64_t local_m = local_block % m_block;
    const int64_t local_index_skew_period = (
        m_block / GreatestCommonDivisor(m_block, n_block) * n_block);
    const int64_t local_n = (
        local_block + local_block / local_index_skew_period) % n_block;
    const int64_t m_region = region / n_regions;
    const int64_t n_position = region % n_regions;
    const int64_t n_region = (m_region & 1) != 0
        ? n_regions - n_position - 1 : n_position;
    return {
        m_region * m_block + local_m,
        n_region * n_block + local_n,
    };
}

}  // namespace GemmRegionSchedule

#endif  // AIC_MTE2_GEMM_REGION_SCHEDULE_H

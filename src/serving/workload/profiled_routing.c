#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int midlayer_profiled_weighted_counts(const double *weights,
        size_t expert_count, size_t tokens, size_t topk,
        const double *ratios, uint64_t *counts) {
    size_t *available = malloc(expert_count * sizeof(*available));
    if (available == NULL) return -1;
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t expert = 0; expert < expert_count; ++expert) available[expert] = expert;
        size_t remaining = expert_count;
        for (size_t route = 0; route < topk; ++route) {
            double total = 0.0;
            for (size_t index = 0; index < remaining; ++index) total += weights[available[index]];
            if (!isfinite(total) || total <= 0.0) { free(available); return -2; }
            double target = ratios[token * topk + route] * total;
            double cumulative = 0.0;
            size_t selected = remaining - 1;
            for (size_t index = 0; index < remaining; ++index) {
                cumulative += weights[available[index]];
                if (target < cumulative) { selected = index; break; }
            }
            ++counts[available[selected]];
            memmove(available + selected, available + selected + 1,
                (remaining - selected - 1) * sizeof(*available));
            --remaining;
        }
    }
    free(available);
    return 0;
}

// Independent register-stream mechanisms. No sorting, reduction, argmax,
// routing, model tensors, or timed global-memory accesses.
// CPU_ONLY produces a portable reference/self-check executable, without CUDA.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MIDLAYER_TYPED_CPU_ONLY
#define HD
#else
#include <cuda_runtime.h>
#define HD __host__ __device__
#endif

enum Kind { F32_GT, F32_EQ, F32_NAN, S64_LT, U64_GT, SHUFFLE32
#ifdef MIDLAYER_TYPED_A800
    , PRMT_ROTATE16, PRMT_BF16_EXPAND, SHUFFLE_DOWN32_1_2_4_8_16
#endif
};
#ifdef MIDLAYER_TYPED_A800
constexpr int kLastKind = SHUFFLE_DOWN32_1_2_4_8_16;
constexpr unsigned kExpectedSms = 108;
#else
constexpr int kLastKind = SHUFFLE32;
constexpr unsigned kExpectedSms = 132;
#endif
constexpr int kUnroll = 31;  // Odd length makes predicate outcomes observable.
const char* names[] = {"f32_gt", "f32_eq", "f32_nan", "s64_lt", "u64_gt", "shuffle32"
#ifdef MIDLAYER_TYPED_A800
    , "prmt_rotate16", "prmt_bf16_expand", "shuffle_down32_1_2_4_8_16"
#endif
};

HD float as_float(std::uint32_t v) { union { std::uint32_t u; float f; } x; x.u = v; return x.f; }
HD std::uint32_t as_bits(float v) { union { std::uint32_t u; float f; } x; x.f = v; return x.u; }
std::int64_t as_signed(std::uint64_t v) { std::int64_t x; std::memcpy(&x, &v, sizeof(x)); return x; }

struct Bounds { float fl, fh, fp; std::uint64_t ul, uh, up; };
HD Bounds bounds(Kind kind, unsigned tid) {
    Bounds b;
    b.fl = -static_cast<float>(1 + tid % 7);
    b.fh = kind == F32_NAN ? as_float(0x7fc00001u + tid % 7) : -b.fl;
    b.fp = kind == F32_EQ ? b.fh : 0.0f;
    b.ul = kind == S64_LT ? (0xfffffff000000000ull | (tid % 31)) : (0x00000000ffffff00ull + tid % 31);
    b.uh = kind == S64_LT ? (0x00000001fedcba00ull + tid % 31) : (0xffffffff00000000ull + tid % 31);
    b.up = kind == S64_LT ? 0xfffffffeffffffffull : 0x8000000087654321ull;
#ifdef MIDLAYER_TYPED_A800
    b.up = kind == S64_LT ? 0xfffffffe80000000ull : 0x8000000080000000ull;
    // Both full-width order cases execute: half the lanes share the pivot's
    // high word and must compare low words, half differ in the high word.
    if (tid & 2) { b.ul = b.up - 1; b.uh = b.up + 1; }
#endif
    return b;
}
HD std::uint64_t initial(Kind kind, unsigned tid, int chain) {
    const Bounds b = bounds(kind, tid);
    if (kind == SHUFFLE32
#ifdef MIDLAYER_TYPED_A800
        || kind == SHUFFLE_DOWN32_1_2_4_8_16
#endif
    ) return (tid << 3) | static_cast<unsigned>(chain);
#ifdef MIDLAYER_TYPED_A800
    if (kind == PRMT_ROTATE16 || kind == PRMT_BF16_EXPAND)
        return 0x12345678u ^ (tid * 0x01010101u) ^ (static_cast<unsigned>(chain) * 0x76543210u);
#endif
    const bool high = (tid + chain) & 1;
    if (kind <= F32_NAN) return as_bits(high ? b.fh : b.fl);
    return high ? b.uh : b.ul;
}

std::uint64_t reference_step(Kind kind, std::uint64_t value, Bounds b) {
    const float f = as_float(static_cast<std::uint32_t>(value));
    switch (kind) {
        case F32_GT: return as_bits(f > b.fp ? b.fl : b.fh);
        case F32_EQ: return as_bits(f == b.fp ? b.fl : b.fh);
        case F32_NAN: return as_bits(std::isnan(f) ? b.fl : b.fh);
        case S64_LT: return as_signed(value) < as_signed(b.up) ? b.uh : b.ul;
        case U64_GT: return value > b.up ? b.ul : b.uh;
        case SHUFFLE32: throw std::runtime_error("shuffle is a warp operation");
#ifdef MIDLAYER_TYPED_A800
        case PRMT_ROTATE16: {
            const auto v = static_cast<std::uint32_t>(value);
            return (v << 16) | (v >> 16);
        }
        case PRMT_BF16_EXPAND: return static_cast<std::uint32_t>(value) << 16;
        case SHUFFLE_DOWN32_1_2_4_8_16: throw std::runtime_error("shuffle down is a warp operation");
#endif
    }
    throw std::runtime_error("invalid primitive");
}

// The original six streams are period two; verify that fact on CPU before use.
// The A800 bit-expansion stream has its explicit collapsing domain below.
// This bounds host validation cost independently of GPU loop length.
std::uint64_t expected(Kind kind, unsigned tid, int chain, unsigned steps) {
    if (kind == SHUFFLE32) return initial(kind, tid ^ (steps & 1), chain);
#ifdef MIDLAYER_TYPED_A800
    // One complete 1/2/4/8/16 down chain maps every destination lane to
    // the original lane31 in its own warp. Out-of-domain sources retain the
    // current destination value at each stage, matching PTX clamp31.
    if (kind == SHUFFLE_DOWN32_1_2_4_8_16)
        return initial(kind, steps == 0 ? tid : ((tid & ~31u) | 31u), chain);
    // The exact BF16 bit-expansion recurrence is NOT period two: it reaches
    // zero after two steps. Preserve that domain; do not claim live source bits.
    if (kind == PRMT_BF16_EXPAND) {
        if (steps == 0) return initial(kind, tid, chain);
        if (steps == 1) return static_cast<std::uint32_t>(initial(kind, tid, chain)) << 16;
        return 0;
    }
#endif
    const auto x = initial(kind, tid, chain);
    const Bounds b = bounds(kind, tid);
    if (reference_step(kind, reference_step(kind, x, b), b) != x)
        throw std::runtime_error("synthetic stream is not period two");
    return steps & 1 ? reference_step(kind, x, b) : x;
}

int self_test() {
    unsigned checks = 0;
    for (int k = F32_GT; k <= kLastKind; ++k) {
        const Kind kind = static_cast<Kind>(k);
        for (int chain = 0; chain < 4; ++chain) {
            for (unsigned steps : {0u, 1u, 2u, 31u, 32u, 129u}) {
                std::uint64_t lanes[32];
                for (unsigned lane = 0; lane < 32; ++lane) lanes[lane] = initial(kind, lane, chain);
                for (unsigned step = 0; step < steps; ++step) {
                    std::uint64_t next[32];
                    if (kind == SHUFFLE32) {
                        for (unsigned lane = 0; lane < 32; ++lane) next[lane] = lanes[lane ^ 1];
#ifdef MIDLAYER_TYPED_A800
                    } else if (kind == SHUFFLE_DOWN32_1_2_4_8_16) {
                        for (unsigned offset : {1u, 2u, 4u, 8u, 16u}) {
                            for (unsigned lane = 0; lane < 32; ++lane)
                                next[lane] = lane + offset < 32 ? lanes[lane + offset] : lanes[lane];
                            std::memcpy(lanes, next, sizeof(lanes));
                        }
#endif
                    } else {
                        for (unsigned lane = 0; lane < 32; ++lane) next[lane] = reference_step(kind, lanes[lane], bounds(kind, lane));
                    }
                    std::memcpy(lanes, next, sizeof(lanes));
                }
                for (unsigned lane = 0; lane < 32; ++lane) {
                    if (lanes[lane] != expected(kind, lane, chain, steps)) return 2;
                    ++checks;
                }
            }
        }
    }
    Bounds b = bounds(F32_GT, 0);
    if (reference_step(F32_GT, 0x7fc01234u, b) != as_bits(b.fh)) return 3;
    ++checks;
    b = bounds(F32_EQ, 0); b.fp = 0.0f;
    if (reference_step(F32_EQ, 0x80000000u, b) != as_bits(b.fl)) return 3;
    ++checks;
    b = bounds(F32_NAN, 0);
    if (reference_step(F32_NAN, 0x7fc01234u, b) != as_bits(b.fl)) return 3;
    ++checks;
    b = bounds(S64_LT, 0);
    if (reference_step(S64_LT, 0x8000000000000000ull, b) != b.uh) return 3;
    ++checks;
    b = bounds(U64_GT, 0);
    if (reference_step(U64_GT, 0xffffffffffffffffull, b) != b.ul) return 3;
    ++checks;
    std::printf("{\"schema\":\"typed_register_cpu_reference_v1\",\"checks\":%u,\"passed\":true,\"GPU_executed\":false,\"inline_PTX_executed\":false}\n", checks);
    return 0;
}

#ifndef MIDLAYER_TYPED_CPU_ONLY
struct Record { unsigned long long begin, end; unsigned smid, smid_space; };

template<int K> __device__ __forceinline__ void primitive(float& f, std::uint64_t& u, const Bounds& b) {
    if constexpr (K == F32_GT)
        asm volatile("{.reg .pred p; setp.gt.f32 p,%1,%4; selp.f32 %0,%2,%3,p;}" : "=f"(f) : "f"(f), "f"(b.fl), "f"(b.fh), "f"(b.fp));
    else if constexpr (K == F32_EQ)
        asm volatile("{.reg .pred p; setp.eq.f32 p,%1,%4; selp.f32 %0,%2,%3,p;}" : "=f"(f) : "f"(f), "f"(b.fl), "f"(b.fh), "f"(b.fp));
    else if constexpr (K == F32_NAN)
        asm volatile("{.reg .pred p; setp.nan.f32 p,%1,%1; selp.f32 %0,%2,%3,p;}" : "=f"(f) : "f"(f), "f"(b.fl), "f"(b.fh));
    else if constexpr (K == S64_LT)
        asm volatile("{.reg .pred p; setp.lt.s64 p,%1,%4; selp.b64 %0,%2,%3,p;}" : "=l"(u) : "l"(u), "l"(b.uh), "l"(b.ul), "l"(b.up));
    else if constexpr (K == U64_GT)
        asm volatile("{.reg .pred p; setp.gt.u64 p,%1,%4; selp.b64 %0,%2,%3,p;}" : "=l"(u) : "l"(u), "l"(b.ul), "l"(b.uh), "l"(b.up));
#ifdef MIDLAYER_TYPED_A800
    else if constexpr (K == PRMT_ROTATE16) {
        unsigned v = static_cast<unsigned>(u);
        asm volatile("prmt.b32 %0,%1,0,0x1032;" : "=r"(v) : "r"(v));
        u = v;
    } else if constexpr (K == PRMT_BF16_EXPAND) {
        unsigned v = static_cast<unsigned>(u);
        asm volatile("prmt.b32 %0,0,%1,0x5410;" : "=r"(v) : "r"(v));
        u = v;
    } else if constexpr (K == SHUFFLE_DOWN32_1_2_4_8_16) {
        unsigned v = static_cast<unsigned>(u);
        asm volatile(
            "shfl.sync.down.b32 %0,%0,1,31,0xffffffff;\n\t"
            "shfl.sync.down.b32 %0,%0,2,31,0xffffffff;\n\t"
            "shfl.sync.down.b32 %0,%0,4,31,0xffffffff;\n\t"
            "shfl.sync.down.b32 %0,%0,8,31,0xffffffff;\n\t"
            "shfl.sync.down.b32 %0,%0,16,31,0xffffffff;"
            : "+r"(v));
        u = v;
    }
#endif
    else {
        unsigned v = static_cast<unsigned>(u);
        asm volatile("shfl.sync.bfly.b32 %0,%1,1,31,0xffffffff;" : "=r"(v) : "r"(v));
        u = v;
    }
}

template<int K, int Chains> __global__ void probe(Record* records, std::uint64_t* values, int outer, std::uint64_t wide_pivot) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    Bounds b = bounds(static_cast<Kind>(K), tid);
    b.up = wide_pivot;  // Runtime operand: do not specialize s64 less-than to a sign test.
    float f[Chains]; std::uint64_t u[Chains];
    #pragma unroll
    for (int c = 0; c < Chains; ++c) {
        u[c] = initial(static_cast<Kind>(K), tid, c);
        f[c] = as_float(static_cast<unsigned>(u[c]));
    }
    __syncthreads();
    unsigned long long begin, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin) :: "memory");
    #pragma unroll 1
    for (int i = 0; i < outer; ++i) {
        #pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
            #pragma unroll
            for (int c = 0; c < Chains; ++c) primitive<K>(f[c], u[c], b);
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end) :: "memory");
    if (threadIdx.x % 32 == 0) {
        unsigned smid, smid_space;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        asm volatile("mov.u32 %0, %%nsmid;" : "=r"(smid_space));
        records[tid / 32] = {begin, end, smid, smid_space};
    }
    #pragma unroll
    for (int c = 0; c < Chains; ++c) values[tid * Chains + c] = K <= F32_NAN ? as_bits(f[c]) : u[c];
}

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<int K> void launch(int chains, int threads, int grid, Record* records, std::uint64_t* values, int outer) {
    const auto pivot = bounds(static_cast<Kind>(K), 0).up;
    if (chains == 1) probe<K, 1><<<grid, threads>>>(records, values, outer, pivot);
    else probe<K, 4><<<grid, threads>>>(records, values, outer, pivot);
}
void run_gpu(Kind kind, int chains, int threads, int grid, int outer, int samples) {
    cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop, 0));
#ifdef MIDLAYER_TYPED_A800
    if (prop.major != 8 || prop.minor != 0 || prop.multiProcessorCount != kExpectedSms || !std::strstr(prop.name, "A800"))
        throw std::runtime_error("requires owned full A800 SM80/108-SM device");
    char pci_bus_id[32]; check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
    std::fprintf(stderr, "a800_cuda_device_identity,%s,%s,%d,%d,%d\n",
                 prop.name, pci_bus_id, prop.major, prop.minor, prop.multiProcessorCount);
#else
    if (prop.major != 9 || prop.minor != 0 || prop.multiProcessorCount != 132 || !std::strstr(prop.name, "H200"))
        throw std::runtime_error("requires owned full H200 SM90/132-SM device");
#endif
    const unsigned count = grid * threads;
    Record* records; std::uint64_t* values;
    check(cudaMalloc(&records, count / 32 * sizeof(Record)));
    check(cudaMalloc(&values, count * chains * sizeof(std::uint64_t)));
    std::vector<Record> host_records(count / 32);
    std::vector<std::uint64_t> host_values(count * chains);
    auto invoke = [&] {
        switch (kind) {
            case F32_GT: launch<F32_GT>(chains, threads, grid, records, values, outer); break;
            case F32_EQ: launch<F32_EQ>(chains, threads, grid, records, values, outer); break;
            case F32_NAN: launch<F32_NAN>(chains, threads, grid, records, values, outer); break;
            case S64_LT: launch<S64_LT>(chains, threads, grid, records, values, outer); break;
            case U64_GT: launch<U64_GT>(chains, threads, grid, records, values, outer); break;
            case SHUFFLE32: launch<SHUFFLE32>(chains, threads, grid, records, values, outer); break;
#ifdef MIDLAYER_TYPED_A800
            case PRMT_ROTATE16: launch<PRMT_ROTATE16>(chains, threads, grid, records, values, outer); break;
            case PRMT_BF16_EXPAND: launch<PRMT_BF16_EXPAND>(chains, threads, grid, records, values, outer); break;
            case SHUFFLE_DOWN32_1_2_4_8_16: launch<SHUFFLE_DOWN32_1_2_4_8_16>(chains, threads, grid, records, values, outer); break;
#endif
        }
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
    };
    invoke();  // Untimed warmup, still no production operator.
    std::printf("sample,primitive,chains,threads,grid,outer,warp_index,smid,smid_space,begin_cycle,end_cycle,warp_bundles,checksum,validation_status\n");
    for (int sample = 0; sample < samples; ++sample) {
        invoke();
        check(cudaMemcpy(host_records.data(), records, host_records.size() * sizeof(Record), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_values.data(), values, host_values.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        for (unsigned tid = 0; tid < count; ++tid)
            for (int c = 0; c < chains; ++c)
                if (host_values[tid * chains + c] != expected(kind, tid, c, outer * kUnroll))
                    throw std::runtime_error("GPU result differs from synthetic CPU reference");
        for (unsigned w = 0; w < host_records.size(); ++w) {
            const auto& r = host_records[w];
            if (r.end <= r.begin || r.smid_space < kExpectedSms || r.smid >= r.smid_space) throw std::runtime_error("invalid same-SM clock interval");
            unsigned long long checksum = 0;
            for (unsigned tid = 32 * w; tid < 32 * (w + 1); ++tid)
                for (int c = 0; c < chains; ++c) checksum += host_values[tid * chains + c];
            std::printf("%d,%s,%d,%d,%d,%d,%u,%u,%u,%llu,%llu,%llu,%llu,pass\n", sample, names[kind], chains, threads, grid, outer, w, r.smid, r.smid_space, r.begin, r.end, static_cast<unsigned long long>(outer) * kUnroll * chains, checksum);
        }
    }
    check(cudaFree(values)); check(cudaFree(records));
}
#endif

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
#ifndef MIDLAYER_TYPED_CPU_ONLY
        if (argc != 8 || std::string(argv[1]) != "--sample") throw std::runtime_error("expected --sample primitive chains threads grid outer samples");
        Kind kind = F32_GT; bool found = false;
        for (int k = F32_GT; k <= kLastKind; ++k) if (argv[2] == std::string(names[k])) { kind = static_cast<Kind>(k); found = true; }
        if (!found) throw std::runtime_error("unknown primitive");
        int args[5];
        for (int i = 0; i < 5; ++i) { std::size_t end; args[i] = std::stoi(argv[i + 3], &end); if (end != std::strlen(argv[i + 3])) throw std::runtime_error("nonintegral argument"); }
        if ((args[0] != 1 && args[0] != 4) || args[1] < 32 || args[1] > 256 || args[1] % 32 || args[2] < 1 || args[2] > 528 || args[3] < 1 || args[3] > 2047 || args[4] < 1 || args[4] > 10)
            throw std::runtime_error("argument outside declared mechanism domain");
        run_gpu(kind, args[0], args[1], args[2], args[3], args[4]); return 0;
#else
        throw std::runtime_error("CPU-only reference accepts --self-test; GPU execution is absent");
#endif
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
